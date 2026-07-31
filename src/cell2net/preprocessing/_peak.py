"""Functions to process scATAC-seq peaks"""
import os
import re

import pandas as pd
import numpy as np
import pyfaidx
from mudata import MuData
from tqdm import tqdm

from cell2net._logging import logger


def get_chrom_sizes(ref_fasta: str) -> dict[str, int]:
    """
    Read chromosome sizes from an indexed FASTA file.
 
    Prefers reading the ``.fai`` index directly (fast, no file handles kept
    open). Falls back to ``pyfaidx``, which will create the index if missing.
 
    Parameters
    ----------
    ref_fasta :
        Path to the reference FASTA file.
 
    Returns
    -------
    dict
        Mapping of chromosome name to length in base pairs.
    """
    fai_path = f"{ref_fasta}.fai"
    if os.path.exists(fai_path):
        df_fai = pd.read_csv(
            fai_path,
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["chrom", "length"],
            dtype={0: str},
        )
        return dict(zip(df_fai["chrom"].astype(str), df_fai["length"].astype(int)))
 
    logger.info(f"No .fai index found for {ref_fasta}, building one with pyfaidx")
    pyf = pyfaidx.Fasta(ref_fasta)
    return {str(name): len(seq) for name, seq in pyf.items()}


def _sanitize_intervals(
    df: pd.DataFrame,
    require_strand: bool = False,
) -> pd.DataFrame:
    """
    Normalize an interval DataFrame so downstream numpy code behaves.
 
    Casts ``Chromosome`` (and ``Strand``) out of ``category`` dtype, coerces
    coordinates to int, and drops rows with missing / unusable values.
 
    Categorical dtypes are the source of most pyranges-era breakage: an unused
    category (e.g. a ``"."`` strand level with zero rows) produces phantom
    empty groups during ``groupby``. Casting to ``str`` sidesteps this
    entirely.
    """
    df = df.copy()
    df["Chromosome"] = df["Chromosome"].astype(str)
 
    n_before = len(df)
    df = df.dropna(subset=["Chromosome", "Start", "End"])
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
 
    if require_strand:
        df["Strand"] = df["Strand"].astype(str)
        bad = ~df["Strand"].isin(["+", "-"])
        if bad.any():
            logger.warning(
                f"Dropping {int(bad.sum())} intervals with unusable strand "
                f"(values: {sorted(df.loc[bad, 'Strand'].unique())})"
            )
            df = df[~bad]
 
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} interval(s) with missing/invalid values")
 
    return df.reset_index(drop=True)


def extend_intervals(
    df: pd.DataFrame,
    up_stream: int = 0,
    down_stream: int = 0,
) -> pd.DataFrame:
    """
    Extend stranded intervals in the 5' and 3' directions.
 
    Replacement for ``pyranges.PyRanges.extend({"5": ..., "3": ...})``.
 
    For ``+`` strand features the 5' end is ``Start`` and the 3' end is ``End``;
    for ``-`` strand features this is reversed. Coordinates are clamped at 0
    (use :func:`clip_to_genome` to clamp the upper bound).
 
    Parameters
    ----------
    df :
        DataFrame with ``Chromosome``, ``Start``, ``End``, ``Strand`` columns.
    up_stream :
        Number of base pairs to extend in the 5' direction.
    down_stream :
        Number of base pairs to extend in the 3' direction.
 
    Returns
    -------
    pd.DataFrame
        A new DataFrame with extended coordinates.
 
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "Chromosome": ["chr1", "chr1"],
    ...     "Start": [1000, 1000],
    ...     "End": [1001, 1001],
    ...     "Strand": ["+", "-"],
    ... })
    >>> extend_intervals(df, up_stream=100, down_stream=50)[["Start", "End"]]
       Start   End
    0    900  1051
    1    950  1101
    """
    df = df.copy()
    is_plus = (df["Strand"].astype(str) == "+").to_numpy()
 
    start = df["Start"].to_numpy(dtype=np.int64).copy()
    end = df["End"].to_numpy(dtype=np.int64).copy()
 
    # 5' extension
    start[is_plus] -= up_stream
    end[~is_plus] += up_stream
 
    # 3' extension
    end[is_plus] += down_stream
    start[~is_plus] -= down_stream
 
    df["Start"] = np.maximum(start, 0)
    df["End"] = end
    return df

def clip_to_genome(
    df: pd.DataFrame,
    chrom_sizes: dict[str, int],
    drop_unknown_chroms: bool = True,
) -> pd.DataFrame:
    """
    Clip intervals to valid genome bounds.
 
    Replacement for ``pyranges.genomicfeatures.genome_bounds(..., clip=True)``.
 
    Parameters
    ----------
    df :
        DataFrame with ``Chromosome``, ``Start``, ``End`` columns.
    chrom_sizes :
        Mapping of chromosome name to length, e.g. from :func:`get_chrom_sizes`.
    drop_unknown_chroms :
        If True, drop intervals on chromosomes absent from ``chrom_sizes``.
        If False, such intervals are left untouched.
 
    Returns
    -------
    pd.DataFrame
        Intervals clipped to ``[0, chrom_length]``. Intervals that end up
        empty (``Start >= End``) are removed.
    """
    df = df.copy()
    df["Chromosome"] = df["Chromosome"].astype(str)
 
    known = df["Chromosome"].isin(chrom_sizes.keys())
    if not known.all():
        missing = sorted(df.loc[~known, "Chromosome"].unique())
        if drop_unknown_chroms:
            logger.warning(
                f"Dropping {int((~known).sum())} intervals on {len(missing)} "
                f"chromosome(s) not present in the reference: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}"
            )
            df = df[known].reset_index(drop=True)
        else:
            logger.warning(
                f"{len(missing)} chromosome(s) not found in the reference, "
                f"leaving those intervals unclipped: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}"
            )
 
    if df.empty:
        return df
 
    limits = df["Chromosome"].map(chrom_sizes)
    # Chromosomes not in chrom_sizes (when drop_unknown_chroms=False) get no limit
    limits = limits.fillna(np.iinfo(np.int64).max).astype(np.int64)
 
    df["Start"] = np.maximum(df["Start"].to_numpy(dtype=np.int64), 0)
    df["End"] = np.minimum(df["End"].to_numpy(dtype=np.int64), limits.to_numpy())
 
    empty = df["Start"] >= df["End"]
    if empty.any():
        logger.warning(f"Dropping {int(empty.sum())} interval(s) empty after clipping")
        df = df[~empty]
 
    return df.reset_index(drop=True)
 
 
def find_overlaps(
    df_query: pd.DataFrame,
    df_subject: pd.DataFrame,
    query_id: str,
    subject_id: str,
) -> pd.DataFrame:
    """
    Find all overlapping interval pairs between two sets of intervals.
 
    Replacement for ``pyranges.PyRanges.overlap``. Uses a per-chromosome
    ``searchsorted`` scan, which is O(N log N + K) for K reported hits rather
    than the O(N*M) of a naive nested loop.
 
    Two half-open intervals overlap when ``a.Start < b.End`` and
    ``a.End > b.Start`` on the same chromosome. Strand is ignored.
 
    Parameters
    ----------
    df_query :
        DataFrame with ``Chromosome``, ``Start``, ``End`` and ``query_id``.
    df_subject :
        DataFrame with ``Chromosome``, ``Start``, ``End`` and ``subject_id``.
    query_id :
        Column in ``df_query`` holding a unique identifier per row.
    subject_id :
        Column in ``df_subject`` holding a unique identifier per row.
 
    Returns
    -------
    pd.DataFrame
        Two columns, ``query_id`` and ``subject_id``, one row per overlapping
        pair.
    """
    if df_query.empty or df_subject.empty:
        return pd.DataFrame({query_id: [], subject_id: []})
 
    subject_by_chrom = {
        chrom: g.sort_values("Start")
        for chrom, g in df_subject.groupby("Chromosome", observed=True, sort=False)
    }
 
    q_hits: list[np.ndarray] = []
    s_hits: list[np.ndarray] = []
 
    for chrom, g_query in df_query.groupby("Chromosome", observed=True, sort=False):
        g_subject = subject_by_chrom.get(chrom)
        if g_subject is None or len(g_subject) == 0:
            continue
 
        s_start = g_subject["Start"].to_numpy(dtype=np.int64)
        s_end = g_subject["End"].to_numpy(dtype=np.int64)
        s_ids = g_subject[subject_id].to_numpy()
        max_width = int((s_end - s_start).max())
 
        q_start = g_query["Start"].to_numpy(dtype=np.int64)
        q_end = g_query["End"].to_numpy(dtype=np.int64)
        q_ids = g_query[query_id].to_numpy()
 
        # Any subject that overlaps [q_start, q_end) must start within
        # [q_start - max_width, q_end).
        lo = np.searchsorted(s_start, q_start - max_width, side="left")
        hi = np.searchsorted(s_start, q_end, side="left")
 
        for i in range(len(q_ids)):
            if hi[i] <= lo[i]:
                continue
            window = slice(lo[i], hi[i])
            keep = s_end[window] > q_start[i]
            if not keep.any():
                continue
            matched = s_ids[window][keep]
            s_hits.append(matched)
            q_hits.append(np.full(matched.shape, q_ids[i], dtype=q_ids.dtype))
 
    if not q_hits:
        return pd.DataFrame({query_id: [], subject_id: []})
 
    return pd.DataFrame(
        {
            query_id: np.concatenate(q_hits),
            subject_id: np.concatenate(s_hits),
        }
    )

def add_peaks(
    mdata: MuData,
    mod_name: str = "atac",
    delimiter="-",
    peak_len: int = 256,
) -> None:
    """
    Add peak metadata to an ATAC-seq modality in a MuData object.

    This function parses peak information from variable names in the AnnData object of a
    specified modality within a MuData object.
    It computes the genomic coordinates (chromosome, start, end, and summit) for each peak and adds them as metadata in the `.var`
    attribute of the AnnData object.

    Parameters
    ----------
    mdata :
        A MuData object containing the ATAC-seq modality to be updated.
    mod_name :
        The name of the modality containing the peak data. Defaults to "atac".
    delimiter :
        The delimiter used to split the variable names in the AnnData object. Defaults to "-".
    peak_len :
        The standardized length of the peaks. The midpoint of each peak is computed,
        and the start and end positions are adjusted to match this length. Defaults to 256.
    chr_var_key :
        The key under which chromosome names will be stored in the `.var` attribute. Defaults to "chr".
    start_var_key :
        The key under which the start positions of peaks will be stored in the `.var` attribute. Defaults to "start".
    end_var_key :
        The key under which the end positions of peaks will be stored in the `.var` attribute. Defaults to "end".
    summit_var_key :
        The key under which the summit (midpoint) positions of peaks will be stored in the `.var` attribute. Defaults to "summit".

    Returns
    -------
    None
        The function modifies the MuData object in place by adding the computed peak
        metadata to the `.var` attribute of the specified modality.

    Raises
    ------
    AssertionError
        If the specified modality (`mod_name`) is not found in the MuData object.

    Notes
    -----
        - The variable names in the AnnData object are expected to follow the format `chromosome{delimiter}start{delimiter}end` (e.g., "chr1-100-200").
        - The peak summit is calculated as the midpoint of the start and end positions, and the peak length is standardized to `peak_len`.

    Examples
    --------
    >>> from mudata import MuData
    >>> import anndata as ad
    >>> import pandas as pd
    >>> import cell2net as cn
    >>> data = ad.AnnData(var=pd.DataFrame(index=["chr1-100-200", "chr2-300-400"]))
    >>> mdata = MuData({"atac": data})
    >>> cn.pp.add_peaks(mdata, mod_name="atac", peak_len=256)
    >>> print(mdata["atac"].var)
        chr  start  end  summit
    0   chr1     72  328     150
    1   chr2    272  528     350
    """
    assert mod_name in mdata.mod_names, f"Cannot find modality: {mod_name}"
    adata = mdata[mod_name]

    chrom_list, start_list, end_list, summit_list = [], [], [], []
    for i in range(adata.n_vars):
        peak = re.split(delimiter, adata.var_names[i])
        chrom, start, end = peak[0], int(peak[1]), int(peak[2])

        chrom_list.append(chrom)

        _mid = (start + end) // 2
        _start = _mid - (peak_len // 2)
        _end = _start + peak_len

        start_list.append(_start)
        end_list.append(_end)
        summit_list.append(_mid)

    adata.uns["peaks"] = pd.DataFrame(
        data={
            "chr": chrom_list,
            "start": start_list,
            "end": end_list,
            "summit": summit_list,
        },
        index=adata.var_names,
    )

    # adata.var[chr_var_key] = chrom_list
    # adata.var[start_var_key] = start_list
    # adata.var[end_var_key] = end_list
    # adata.var[summit_var_key] = summit_list

    return None


def peak_to_gene(
    mdata: MuData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    gene_name_col: str = "gene_names",
    up_stream: int = 500_000,
    down_stream: int = 500_000,
    ref_fasta: str = "",
    highly_variable: bool = False,
    genes: list[str] | None = None,
    min_n_peaks: int = 1,
    max_pct_dropout_by_counts: float | None = None,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Link peaks to genes based on proximity to transcription start sites (TSS).

    This function assigns ATAC-seq peaks to genes based on their proximity to
    transcription start sites (TSS) within specified upstream and downstream
    distances. The resulting mapping can be added to the `uns` attribute of the
    provided MuData object or returned as a DataFrame.

    Parameters
    ----------
    mdata :
        A MuData object containing RNA and ATAC modalities.
    rna_mod :
        Name of the RNA modality in the MuData object.
    atac_mod :
        Name of the ATAC modality in the MuData object.
    gene_name_col :
        Column name in the RNA `.var` attribute that contains gene names.
    up_stream :
        Distance upstream of the TSS to consider for assigning peaks.
    down_stream :
        Distance downstream of the TSS to consider for assigning peaks.
    ref_fasta :
        Path to the reference FASTA file for determining genome bounds.
        The file must be indexed.
    chr_var_key :
        Key in ATAC `.var` that contains chromosome names.
    start_var_key :
        Key in ATAC `.var` that contains peak start positions.
    end_var_key :
        Key in ATAC `.var` that contains peak end positions.
    highly_variable :
        If True, only consider highly variable genes.
    genes :
        Specific genes to include in the mapping. If None, all genes are considered.
    min_n_peaks :
        Minimum number of associated peaks required for a gene to be included.
    max_pct_dropout_by_counts :
        Maximum percentage of dropout by counts for filtering genes.
        If None, no filtering is applied and all genes are used.
    inplace :
        If True, the resulting mapping is added to the `uns` attribute of the MuData object under the key "peak_to_gene".
        If False, the mapping is returned as a DataFrame.

    Returns
    -------
    If `inplace` is False, returns a DataFrame with columns:

        - `gene`: Gene name.
        - `peak`: Peak identifier.
        - `distance`: Distance from the TSS to the peak summit.

    Otherwise, modifies the MuData object in place.

    Raises
    ------
    AssertionError
        If gene TSS coordinates are not found in the RNA modality (`adata_rna.uns["gene_tss_coord"]`).

    Notes
    -----
    - Peaks are assigned to genes based on overlap with genomic regions defined by the upstream and downstream distances from the TSS.
    - Genes without any associated peaks are excluded from the output.
    - Peak summits are calculated as the midpoint of their start and end positions.

    Examples
    --------
    >>> from mudata import MuData
    >>> import anndata as ad
    >>> import pandas as pd
    >>> import cell2net as cn
    >>> mdata = MuData({
    ...     "rna": ad.AnnData(var=pd.DataFrame({"gene_names": ["gene1", "gene2"]})),
    ...     "atac": ad.AnnData(var=pd.DataFrame({
    ...         "chr": ["chr1", "chr1"],
    ...         "start": [100, 200],
    ...         "end": [150, 250]
    ...     }))
    ... })
    >>> mdata["rna"].uns["gene_tss_coord"] = pd.DataFrame({
    ...     "gene_name": ["gene1", "gene2"],
    ...     "tss": [125, 225],
    ...     "strand": ["+", "-"],
    ...     "chrom": ["chr1", "chr1"]
    ... })
    >>> df = peak_to_gene(mdata, ref_fasta="genome.fa", inplace=False)
    >>> print(df.head())
        gene  peak  distance
    0   gene1     0        25
    1   gene2     1        25
    """
    # Check if can find TSS coordinates in adata_rna
    adata_rna = mdata[rna_mod]
    adata_atac = mdata[atac_mod]

    if "gene_tss_coord" not in adata_rna.uns:
        logger.error("Cannot find gene TSS coordinates in adata_rna.uns['gene_tss_coord'], please run `cn.pp.add_gene_tss_coord()` first")
        return None

    if gene_name_col not in adata_rna.var.columns:
        logger.error(f"Cannot find {gene_name_col} in mdata[{rna_mod}].var")
        return None

    logger.info("Fetching TSS coordinates")

    df_tss = adata_rna.uns["gene_tss_coord"].copy()
 
    df_tss = pd.DataFrame(
        {
            "Chromosome": df_tss["chrom"].astype(str),
            "Start": df_tss["tss"].astype(np.int64) - 1,
            "End": df_tss["tss"].astype(np.int64),
            "Name": df_tss["gene_name"].astype(str),
            "Score": 0,
            "Strand": df_tss["strand"].astype(str),
            "tss": df_tss["tss"].astype(np.int64),
        }
    )

    # df_tss = adata_rna.uns["gene_tss_coord"]
    # df_tss["Start"] = df_tss["tss"] - 1
    # df_tss["End"] = df_tss["tss"]
    # df_tss["Score"] = 0
    # df_tss["Name"] = df_tss["gene_name"]
    # df_tss["Strand"] = df_tss["strand"]
    # df_tss["Chromosome"] = df_tss["chrom"]
    # df_tss = df_tss[["Chromosome", "Start", "End", "Name", "Score", "Strand", "tss"]]

    df_var = adata_rna.var

    if highly_variable:
        logger.info("Using highly variable genes")
        df_var = df_var[df_var["highly_variable"]]
        df_tss = df_tss[df_tss["Name"].isin(df_var[gene_name_col])]

    if max_pct_dropout_by_counts is not None:
        logger.info(
            f"Filter genes by pct_dropout_by_counts: {max_pct_dropout_by_counts}"
        )
        df_var = df_var[df_var["pct_dropout_by_counts"] < max_pct_dropout_by_counts]
        df_tss = df_tss[df_tss["Name"].isin(df_var[gene_name_col])]

    if genes is not None:
        df_tss = df_tss[df_tss["Name"].isin(genes)]

    df_tss = _sanitize_intervals(df_tss, require_strand=True)
    if df_tss.empty:
        logger.error("df_tss is empty after removing invalid coordinates!")
        return None


    # ---- Expand TSS into search windows ------------------------------------
    df_windows = extend_intervals(df_tss, up_stream=up_stream, down_stream=down_stream)
 
    chrom_sizes = get_chrom_sizes(ref_fasta)
    df_windows = clip_to_genome(df_windows, chrom_sizes)
    if df_windows.empty:
        logger.error("No gene windows left after clipping to genome bounds!")
        return None
 
    df_windows = df_windows.reset_index(drop=True)
    df_windows["_win_id"] = np.arange(len(df_windows), dtype=np.int64)
    logger.info(f"Number of genes: {df_windows['Name'].nunique()}")

    # gr_genes = pr.from_dict(df_tss)
    # gr_genes = gr_genes.extend({"5": up_stream})
    # gr_genes = gr_genes.extend({"3": down_stream})
    # logger.info(f"Number of genes: {len(gr_genes)}")

    # ---- Peaks --------------------------------------------------------------
    df_peaks_uns = adata_atac.uns["peaks"]
    df_peaks = pd.DataFrame(
        {
            "Chromosome": df_peaks_uns["chr"].astype(str).to_numpy(),
            "Start": df_peaks_uns["start"].to_numpy(dtype=np.int64),
            "End": df_peaks_uns["end"].to_numpy(dtype=np.int64),
            "Peaks": np.asarray(df_peaks_uns.index, dtype=object),
            "Summit": df_peaks_uns["summit"].to_numpy(dtype=np.int64),
        }
    )
    df_peaks = _sanitize_intervals(df_peaks)
    df_peaks["_peak_id"] = np.arange(len(df_peaks), dtype=np.int64)
    logger.info(f"Number of peaks: {len(df_peaks)}")
 
    shared = set(df_windows["Chromosome"]) & set(df_peaks["Chromosome"])
    if not shared:
        logger.error(
            "Gene windows and peaks share no chromosomes — check for a "
            "'chr' prefix mismatch between the TSS annotation and the peak names. "
            f"Genes: {sorted(set(df_windows['Chromosome']))[:5]}, "
            f"Peaks: {sorted(set(df_peaks['Chromosome']))[:5]}"
        )
        return None
 
    # ---- Overlap ------------------------------------------------------------
    logger.info("Linking genes to nearby peaks")
    df_hits = find_overlaps(
        df_windows,
        df_peaks,
        query_id="_win_id",
        subject_id="_peak_id",
    )
 
    if df_hits.empty:
        logger.error("No peak-to-gene links found!")
        return None
 
    df = df_hits.merge(
        df_windows[["_win_id", "Name", "tss"]], on="_win_id", how="left"
    ).merge(
        df_peaks[["_peak_id", "Peaks", "Summit"]], on="_peak_id", how="left"
    )
 
    df["distance"] = (df["Summit"] - df["tss"]).abs()
    df = df.rename(columns={"Name": "gene", "Peaks": "peak"})[
        ["gene", "peak", "distance"]
    ]
 
    # Collapse duplicate gene-peak pairs (genes with several annotated TSS),
    # keeping the shortest distance.
    df = (
        df.sort_values("distance")
        .drop_duplicates(subset=["gene", "peak"], keep="first")
        .reset_index(drop=True)
    )
 
    n_genes_wo_peak = df_windows["Name"].nunique() - df["gene"].nunique()
    if n_genes_wo_peak:
        logger.info(f"Number of genes without any linked peak: {n_genes_wo_peak}")
 
    # ---- Filter by number of linked peaks -----------------------------------
    peaks_per_gene = df.groupby("gene")["peak"].size()
    keep_genes = peaks_per_gene[peaks_per_gene >= min_n_peaks].index
    df = df[df["gene"].isin(keep_genes)].reset_index(drop=True)
 
    df = df.sort_values(["gene", "distance"]).reset_index(drop=True)
 
    logger.info(
        f"Number of genes that have at least {min_n_peaks} peaks: {df['gene'].nunique()}"
    )
    logger.info(f"Identified {len(df)} potential peak-to-gene links")
 
    if inplace:
        mdata.uns["peak_to_gene"] = df
        return None
    return df

    # pyf = pyfaidx.Fasta(ref_fasta)
    # gr_genes = gf.genome_bounds(gr_genes, chromsizes=pyf, clip=True)

    # if "peaks" not in adata_atac.uns:
    #     logger.error("Cannot find peaks in adata_atac.uns['peaks']")
    #     return None

    # df_peaks = pd.DataFrame(
    #         data={
    #             "Chromosome": adata_atac.uns["peaks"]['chr'],
    #             "Start": adata_atac.uns["peaks"]['start'],
    #             "End": adata_atac.uns["peaks"]['end'],
    #             "Peaks": adata_atac.uns["peaks"].index,
    #             "Summit": adata_atac.uns["peaks"]['summit'],
    #         }
    #     )
    # gr_peaks = pr.from_dict(df_peaks)

    # logger.info(f"Number of peaks: {len(gr_peaks)}")

    # logger.info("Linking genes to nearby peaks")
    # df_list, genes_wo_peak = [], []
    # for gene in tqdm(gr_genes.Name, total=len(gr_genes.Name), desc="Linking genes to peaks"):
    #     gr_gene = gr_genes[(gr_genes.Name == gene)]

    #     # find overlap peaks
    #     overlap_peaks = gr_peaks.overlap(gr_gene)
    #     if len(overlap_peaks) == 0:
    #         genes_wo_peak.append(gene)
    #         continue

    #     # Compute distance between TSS and peak summit
    #     overlap_peaks.Distance = abs(overlap_peaks.Summit - gr_gene.tss.values[0])
    #     df = overlap_peaks.df.sort_values("Distance")
    #     df["gene"] = gene
    #     df = df[["gene", "Peaks", "Distance"]]
    #     df.columns = ["gene", "peak", "distance"]
    #     df_list.append(df)

    # df = pd.concat(df_list).reset_index(drop=True)

    # # Remove genes with number of associated peaks less than min_n_peaks
    # grouped_df = df.groupby("gene").count()
    # grouped_df = grouped_df[grouped_df["peak"] > min_n_peaks]

    # df = df[df["gene"].isin(grouped_df.index)].reset_index(drop=True)

    # n_genes = len(df["gene"].unique())
    # logger.info(f"Number of genes that have at least {min_n_peaks} peaks: {n_genes}")

    # logger.info(f"Identified {len(df)} potential peak-to-gene links")

    # if inplace:
    #     mdata.uns["peak_to_gene"] = df
    # else:
    #     return df


def random_regions(
    chrom_sizes: dict,
    n_regions: int,
    length: int = 256,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate random genomic regions of a fixed length.
 
    Parameters
    ----------
    chrom_sizes :
        Mapping of chromosome name to length, e.g. from :func:`get_chrom_sizes`.
    n_regions :
        Number of regions to generate.
    length :
        Length of each region in base pairs.
    random_seed :
        Seed for the random number generator.
 
    Returns
    -------
    pd.DataFrame
        Columns ``chrom``, ``start``, ``end``.
 
    Notes
    -----
    Chromosomes are sampled uniformly, not weighted by length. Chromosomes
    shorter than ``length`` are excluded.
    """
    rng = np.random.default_rng(random_seed)
 
    usable = {c: int(n) for c, n in chrom_sizes.items() if int(n) > length}
    if not usable:
        raise ValueError(f"No chromosome is longer than the requested length ({length})")
 
    chroms = np.asarray(list(usable.keys()), dtype=object)
    lengths = np.asarray([usable[c] for c in chroms], dtype=np.int64)
 
    idx = rng.integers(0, len(chroms), size=n_regions)
    max_start = lengths[idx] - length - 1
    start = (rng.random(n_regions) * max_start).astype(np.int64)
 
    return pd.DataFrame(
        {
            "chrom": chroms[idx],
            "start": start,
            "end": start + length,
        }
    )
