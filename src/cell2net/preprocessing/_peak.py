"""Functions to process scATAC-seq peaks"""

import re

import pandas as pd
import pyfaidx
import pyranges as pr
import pyranges.genomicfeatures as gf
from mudata import MuData
from pysam import FastaFile
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.genome import Genome


def add_peaks(
    mdata: MuData,
    mod_name: str = "atac",
    delimiter="-",
    peak_len: int = 256,
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    summit_var_key: str = "summit",
) -> None:
    """
    Add peak metadata to an ATAC-seq modality in a MuData object.

    This function parses peak information from variable names in the AnnData object of a
    specified modality within a MuData object. It computes the genomic coordinates
    (chromosome, start, end, and summit) for each peak and adds them as metadata in the `.var`
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

    adata.var[chr_var_key] = chrom_list
    adata.var[start_var_key] = start_list
    adata.var[end_var_key] = end_list
    adata.var[summit_var_key] = summit_list

    return None


def add_dna_sequence(
    mdata: MuData,
    ref_fasta: str,
    mod_name: str = "atac",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    sequence_var_key: str = "dna_sequence",
) -> None:
    """
    Add sequences to peak metadata in a MuData object.

    This function retrieves DNA sequences for genomic regions specified in the `.var`
    attribute of the AnnData object within a MuData object. The sequences are fetched
    from a reference FASTA file and added as metadata under the specified key.

    Parameters
    ----------
    mdata : MuData
        A MuData object containing the modality with peak metadata.
    ref_fasta : str
        Path to the reference FASTA file. This file must be indexed (e.g., with samtools faidx).
    mod_name : str, optional
        The name of the modality containing peak data. Defaults to "atac".
    chr_var_key : str, optional
        The key in `.var` that contains chromosome names. Defaults to "chr".
    start_var_key : str, optional
        The key in `.var` that contains the start positions of peaks. Defaults to "start".
    end_var_key : str, optional
        The key in `.var` that contains the end positions of peaks. Defaults to "end".
    sequence_var_key : str, optional
        The key under which the retrieved DNA sequences will be stored in `.var`. Defaults to "dna_sequence".

    Returns
    -------
    None
        The function modifies the MuData object in place by adding DNA sequences to the
        specified key in the `.var` attribute.

    Raises
    ------
    AssertionError
        If the specified modality (`mod_name`) is not found in the MuData object.
    FileNotFoundError
        If the `ref_fasta` file does not exist or is not properly indexed.

    Examples
    --------
    >>> from mudata import MuData
    >>> import anndata as ad
    >>> import pandas as pd
    >>> import cell2net as cn
    >>> data = ad.AnnData(var=pd.DataFrame({
    ...     "chr": ["chr1", "chr2"],
    ...     "start": [100, 200],
    ...     "end": [150, 250]
    ... }))
    >>> mdata = MuData({"atac": data})
    >>> cn.pp.add_dna_sequence(mdata, ref_fasta="reference.fasta")
    >>> print(mdata["atac"].var["dna_sequence"])
    0    ATCGTTGAC...
    1    TGGCCAATA...
    """
    assert mod_name in mdata.mod_names, f"Cannot find modality: {mod_name}"
    adata = mdata[mod_name]

    fasta = FastaFile(filename=ref_fasta)
    df = adata.var[[chr_var_key, start_var_key, end_var_key]]

    seqs = []
    for chrom, start, end in tqdm(
        zip(
            df[chr_var_key],
            df[start_var_key],
            df[end_var_key],
            strict=False,
        )
    ):
        seqs.append(fasta.fetch(chrom, start, end).upper())

    adata.var[sequence_var_key] = seqs

    return None


def peak_to_gene(
    mdata: MuData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    gene_name_col: str = "gene_names",
    up_stream: int = 500_000,
    down_stream: int = 500_000,
    ref_fasta: str = "",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    highly_variable: bool = True,
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
        Name of the RNA modality in the MuData object. Defaults to "rna".
    atac_mod :
        Name of the ATAC modality in the MuData object. Defaults to "atac".
    gene_name_col :
        Column name in the RNA `.var` attribute that contains gene names. Defaults to "gene_names".
    up_stream :
        Distance upstream of the TSS to consider for assigning peaks. Defaults to 500,000.
    down_stream :
        Distance downstream of the TSS to consider for assigning peaks. Defaults to 500,000.
    ref_fasta :
        Path to the reference FASTA file for determining genome bounds. The file must be indexed.
    chr_var_key :
        Key in ATAC `.var` that contains chromosome names. Defaults to "chr".
    start_var_key :
        Key in ATAC `.var` that contains peak start positions. Defaults to "start".
    end_var_key :
        Key in ATAC `.var` that contains peak end positions. Defaults to "end".
    highly_variable :
        If True, only consider highly variable genes. Defaults to True.
    genes :
        Specific genes to include in the mapping. If None, all genes are considered. Defaults to None.
    min_n_peaks :
        Minimum number of associated peaks required for a gene to be included. Defaults to 1.
    max_pct_dropout_by_counts :
        Maximum percentage of dropout by counts for filtering genes. If None, no filtering is applied. Defaults to None.
    inplace :
        If True, the resulting mapping is added to the `uns` attribute of the MuData object under the key "peak_to_gene".
        If False, the mapping is returned as a DataFrame. Defaults to True.

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
    - Peaks are assigned to genes based on overlap with genomic regions defined
        by the upstream and downstream distances from the TSS.
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

    assert "gene_tss_coord" in adata_rna.uns, "Cannot find gene TSS coordinates"

    logger.info("Fetch gene coordinates")
    df_tss = adata_rna.uns["gene_tss_coord"]
    df_tss["Start"] = df_tss["tss"] - 1
    df_tss["End"] = df_tss["tss"]
    df_tss["Score"] = 0
    df_tss["Name"] = df_tss["gene_name"]
    df_tss["Strand"] = df_tss["strand"]
    df_tss["Chromosome"] = df_tss["chrom"]
    df_tss = df_tss[["Chromosome", "Start", "End", "Name", "Score", "Strand", "tss"]]

    df_var = adata_rna.var

    if highly_variable:
        logger.info("Using highly variable genes")
        df_var = df_var[df_var["highly_variable"]]
        df_tss = df_tss[df_tss["Name"].isin(df_var[gene_name_col])]

    if max_pct_dropout_by_counts is not None:
        logger.info("Filter genes by pct_dropout_by_counts")
        df_var = df_var[df_var["pct_dropout_by_counts"] < max_pct_dropout_by_counts]
        df_tss = df_tss[df_tss["Name"].isin(df_var[gene_name_col])]

    if genes is not None:
        df_tss = df_tss[df_tss["Name"].isin(genes)]

    gr_genes = pr.from_dict(df_tss)
    gr_genes = gr_genes.extend({"5": up_stream})
    gr_genes = gr_genes.extend({"3": down_stream})

    pyf = pyfaidx.Fasta(ref_fasta)
    gr_genes = gf.genome_bounds(gr_genes, chromsizes=pyf, clip=True)

    logger.info("Find nearby peaks for each gene")
    df_peaks = pd.DataFrame(
        data={
            "Chromosome": adata_atac.var[chr_var_key],
            "Start": adata_atac.var[start_var_key],
            "End": adata_atac.var[end_var_key],
        }
    )

    gr_peaks = pr.from_dict(df_peaks)
    gr_peaks.Peaks = df_peaks.index.values
    gr_peaks.Summit = (gr_peaks.End + gr_peaks.Start) // 2

    df_list, genes_wo_peak = [], []
    for gene in gr_genes.Name:
        gr_gene = gr_genes[(gr_genes.Name == gene)]

        # find overlap peaks
        overlap_peaks = gr_peaks.overlap(gr_gene)
        if len(overlap_peaks) == 0:
            genes_wo_peak.append(gene)
            continue

        # Compute distance between TSS and peak summit
        overlap_peaks.Distance = abs(overlap_peaks.Summit - gr_gene.tss.values[0])
        df = overlap_peaks.df.sort_values("Distance")
        df["gene"] = gene
        df = df[["gene", "Peaks", "Distance"]]
        df.columns = ["gene", "peak", "distance"]
        df_list.append(df)

    df = pd.concat(df_list).reset_index(drop=True)

    # Remove genes with number of associated peaks less than min_n_peaks
    grouped_df = df.groupby("gene").count()
    grouped_df = grouped_df[grouped_df["peak"] > min_n_peaks]

    df = df[df["gene"].isin(grouped_df.index)]

    n_genes = len(df["gene"].unique())
    logger.info(f"Number of genes: {n_genes}")

    if inplace:
        mdata.uns["peak_to_gene"] = df
    else:
        return df


def annotate_peaks(
    mdata: MuData,
    genome: Genome,
    mod_name: str = "atac",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
):
    pass
