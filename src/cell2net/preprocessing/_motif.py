from collections.abc import Iterable
from typing import Literal, get_args

import MOODS.scan
import MOODS.tools
import numpy as np
import pandas as pd
from mudata import MuData
from scipy.sparse import csr_matrix
from tqdm import tqdm

from cell2net._logging import logger

_BACKGROUND = Literal["subject", "genome", "even"]


def get_motifs_from_jaspar(
    release: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: list[str] | None = None,
    all_versions: bool = False,
) -> Iterable | None:
    """
    Fetch transcription factor motifs from the JASPAR database.

    This function retrieves transcription factor binding motifs from the JASPAR database using the `pyjaspar` library.
    It allows filtering by JASPAR release, motif collection, taxonomic group, and version.

    Parameters
    ----------
    release :
        The release version ( e.g. JASPAR2020, JASPAR2024) of the JASPAR database to query.
    collection :
        The collection of motifs to query. Common options include:

            - `"CORE"`: High-quality, manually curated collection.
            - `"UNVALIDATED"`: Computationally predicted motifs.

    tax_group :
        A list of taxonomic groups to filter motifs. For example:

        - `["vertebrates"]`
        - `["plants", "insects"]`

        If `None`, defaults to `["vertebrates"]`.

    all_versions :
        Whether to fetch all versions of each motif. If `True`, retrieves all motif versions;
        otherwise, retrieves only the latest version.

    Returns
    -------
        An iterable of motif objects fetched from the JASPAR database.
        Returns `None` if the `pyjaspar` library is not installed or an error occurs.

    Raises
    ------
    ImportError
        If the `pyjaspar` library is not installed.

    Notes
    -----
    - Requires the `pyjaspar` library to interact with the JASPAR database. Install it via `pip install pyjaspar`.
    - The motifs fetched are represented as objects with attributes like `name`, `matrix_id`, and `counts`, which can be used for downstream analysis.

    Examples
    --------
    Fetch all motifs from the JASPAR2024 CORE collection for vertebrates:

    >>> motifs = get_motifs_from_jaspar(
    ...     release="JASPAR2024",
    ...     collection="CORE",
    ...     tax_group=["vertebrates"],
    ... )
    >>> print(len(motifs))
    """
    # check if JASPAR is installed
    try:
        from pyjaspar import jaspardb
    except ImportError:
        logger.error(
            "pyjaspar is not installed. Please install it first: pip install pyjaspar"
        )
        return None

    if tax_group is None:
        tax_group = ["vertebrates"]

    jdb_obj = jaspardb(release=release)
    motifs = jdb_obj.fetch_motifs(
        collection=collection, tax_group=tax_group, all_versions=all_versions
    )

    logger.info(f"Number of motifs fetched: {len(motifs)}")

    return motifs


def filter_motifs_by_genes(
    motifs: Iterable,
    mdata: MuData,
    rna_mod: str = "rna",
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Filter motifs by matching their names to expressed gene names in the RNA modality.

    This function identifies transcription factor (TF) motifs whose names overlap
    with the expressed genes in the RNA modality of a multimodal `MuData` object.
    The filtered motifs are returned or stored in `mdata.uns["motifs"]`

    Parameters
    ----------
    motifs :
        A collection of motif objects to be filtered.
        Can be obtained using `get_motifs_from_jaspar`.
        Each motif should have:

        - `name`: Motif name (string).
        - `matrix_id`: Unique identifier for the motif.

    mdata :
        A multimodal data object containing at least an RNA modality.
        The RNA modality should have genes stored in `.var_names`.
    rna_mod :
        The key for the RNA modality in `mdata`.
    inplace :
        If `True`, stores the filtered motifs in `mdata.uns["motifs"]`.
        If `False`, returns the filtered DataFrame.

    Returns
    -------
        - If `inplace=True`: Returns `None`. The filtered motifs are stored in `mdata.uns["motifs"]`.
        - If `inplace=False`: Returns a DataFrame with filtered motifs and their corresponding genes. The DataFrame has the following columns:

            - `"motif_name"`: Name of the motif.
            - `"motif_id"`: Unique identifier of the motif.
            - `"gene_name"`: Name of the matching gene.

    Notes
    -----
    - Gene names from the RNA modality are converted to uppercase for case-insensitive matching with motif names.
    - Duplicate motifs are removed based on their uppercased names, retaining the last occurrence.
    - The filtered DataFrame or stored result includes only motifs with names matching gene names.

    Examples
    --------
    Filter motifs by genes and store the results in `mdata.uns`:

    >>> filter_motifs_by_genes(motifs, mdata, rna_mod="rna", inplace=True)
    >>> mdata.uns["motifs"]

    Filter motifs by genes and return the filtered DataFrame:

    >>> df_filtered = filter_motifs_by_genes(motifs, mdata, inplace=False)
    >>> print(df_filtered)

    Access filtered motifs after storing in `mdata`:

    >>> mdata.uns["motifs"]
    >>> mdata.uns["motifs"].head()

    """
    # collect motif names and ids
    motif_names, motif_ids = [], []
    for motif in motifs:
        motif_names.append(motif.name)
        motif_ids.append(motif.matrix_id)

    df_motif = pd.DataFrame(data={"motif_name": motif_names, "motif_id": motif_ids})
    df_motif["motif_name_upper"] = df_motif["motif_name"].str.upper()
    df_motif.drop_duplicates(subset=["motif_name_upper"], keep="last", inplace=True)

    # filter motifs by gene names
    gene_names = mdata[rna_mod].var_names.tolist()
    df_gene = pd.DataFrame(data={"gene_name": gene_names})
    df_gene["gene_name_upper"] = df_gene["gene_name"].str.upper()

    sel_genes = list(
        set(df_gene["gene_name_upper"].values.tolist())
        & set(df_motif["motif_name_upper"].values.tolist())
    )

    df_motif = df_motif[df_motif["motif_name_upper"].isin(sel_genes)].reset_index(
        drop=True
    )
    df_gene = df_gene[df_gene["gene_name_upper"].isin(sel_genes)].reset_index(drop=True)

    assert len(df_motif) == len(df_gene), "Number of motifs and genes are different!"

    df_motif = pd.merge(
        df_motif,
        df_gene,
        left_on="motif_name_upper",
        right_on="gene_name_upper",
        how="inner",
    )
    df_motif = df_motif[["motif_name", "motif_id", "gene_name"]]

    logger.info(f"Number of motifs overlapped with genes: {len(df_motif)}")

    if inplace:
        mdata.uns["motifs"] = df_motif
    else:
        return df_motif


def match_motif(
    mdata: MuData,
    motifs: Iterable,
    atac_mod: str = "atac",
    pseudocounts: float = 0.0001,
    p_value: float = 5e-05,
    background: _BACKGROUND = "even",
    key_added: str = "motif_match",
    save_bed: bool = False,
    bed_file: str = "motif_match.bed",
) -> None:
    """
    Matches transcription factor motifs to accessible DNA sequences and links them with expressed genes.

    This function identifies transcription factor (TF) binding motifs that are relevant to genes expressed in scRNA-seq data.
    It uses accessible DNA sequences from ATAC-seq data and computationally scans for TF motifs using the MOODS library.
    Results are stored as a sparse matrix in the ATAC modality.

    Parameters
    ----------
    mdata :
        Multimodal data object containing both RNA and ATAC modalities.
    motifs :
        A collection of motif objects. Each motif must have attributes `name`, `matrix_id`, and `counts` representing
        the motif's name, unique identifier, and nucleotide frequencies respectively.
    atac_mod :
        Key for the ATAC modality in `mdata`, by default "atac".
        This modality should contain DNA accessibility data and DNA sequences in `.var["dna_sequence"]`.
    pseudocounts :
        Small value added to motif counts to avoid division by zero in log-odds computations.
    p_value :
         P-value threshold for motif matching.
         Lower values result in stricter matches.
    background :
        Background nucleotide distribution for motif scoring. Choices:

        - `"even"`: Assumes uniform nucleotide frequency.
        - `"subject"`: Uses nucleotide frequencies from accessible DNA sequences in the dataset.
        - `"genome"`: (Not implemented) Placeholder for using genome-wide nucleotide frequencies.

        By default, `"even"`.

    key_added :
        Name of the key to store the resulting motif match matrix in `adata_atac.varm`.

    Returns
    -------
    Results are added to the `mdata` object in place:

        - `mdata.uns["motifs"]`: DataFrame with overlapping motif and gene information.
        - `adata_atac.varm[key_added]`: Sparse matrix indicating motif matches for each accessible DNA sequence.

    Raises
    ------
    AssertionError
        If the DNA sequence information (`"dna_sequence"`) is missing in `adata_atac.var`.
    AssertionError
        If the number of motifs does not match the number of overlapping genes after filtering.
    ValueError
        If the `background` parameter is not one of the predefined choices (`"even"`, `"subject"`, or `"genome"`).

    Notes
    -----
    - It computes motif log-odds scores based on the provided background nucleotide frequencies.
    - Motif matching is performed on accessible DNA sequences using the MOODS library, which allows for efficient scanning and p-value thresholding.
    - The resulting sparse matrix is binary (0 or 1), where 1 indicates the presence of a significant motif match.

    Examples
    --------
    Match TF motifs to accessible regions and associate them with expressed genes:

    >>> match_motif(
    ...     mdata,
    ...     motifs=motif_list,
    ...     rna_mod="rna",
    ...     atac_mod="atac",
    ...     pseudocounts=0.0001,
    ...     p_value=5e-05,
    ...     background="even",
    ...     key_added="motif_match",
    ... )

    Access overlapping motifs and genes:

    >>> mdata.uns["motifs"]

    Access motif match results:

    >>> mdata["atac"].varm["motif_match"]

    Customize background nucleotide frequencies:

    >>> match_motif(mdata, motifs=motif_list, background="subject")
    """
    adata_atac = mdata[atac_mod]

    assert (
        "dna_sequence" in adata_atac.var.columns
    ), "Cannot find sequences, please first run cell2net.pp.add_dna_sequence"

    # subset motifs
    motif_ids = mdata.uns["motifs"]["motif_id"].values.tolist()
    motifs_sub = []
    for motif in motifs:
        if motif.matrix_id in motif_ids:
            motifs_sub.append(motif)

    logger.info(f"Number of motifs: {len(motifs_sub)}")

    logger.info("Find TF binding sites")
    # motif matching
    options = get_args(_BACKGROUND)
    assert background in options, f"'{background}' is not in {options}"

    # compute background distribution
    seq = ""
    if background == "subject":
        for i in range(adata_atac.n_vars):
            seq += adata_atac.uns["peak_seq"][i]
        _bg = MOODS.tools.bg_from_sequence_dna(seq, 0)
    elif background == "genome":
        # TODO
        _bg = MOODS.tools.flat_bg(4)
    else:
        _bg = MOODS.tools.flat_bg(4)

    # prepare motif data
    n_motifs = len(motifs_sub)

    matrices = [None] * 2 * n_motifs
    thresholds = [None] * 2 * n_motifs
    for i, motif in enumerate(motifs_sub):
        counts = (
            tuple(motif.counts["A"]),
            tuple(motif.counts["C"]),
            tuple(motif.counts["G"]),
            tuple(motif.counts["T"]),
        )

        matrices[i] = MOODS.tools.log_odds(counts, _bg, pseudocounts)
        matrices[i + n_motifs] = MOODS.tools.reverse_complement(matrices[i])

        thresholds[i] = MOODS.tools.threshold_from_p(matrices[i], _bg, p_value)
        thresholds[i + n_motifs] = thresholds[i]

    # create scanner
    scanner = MOODS.scan.Scanner(7)
    scanner.set_motifs(matrices=matrices, bg=_bg, thresholds=thresholds)
    motif_match = np.zeros(shape=(adata_atac.n_vars, n_motifs), dtype=np.uint8)

    for i in tqdm(range(adata_atac.n_vars)):
        results = scanner.scan(adata_atac.var["dna_sequence"].iloc[i])
        for j in range(n_motifs):
            if len(results[j]) > 0 or len(results[j + n_motifs]) > 0:
                motif_match[i, j] = 1  # type: ignore

    adata_atac.varm[key_added] = csr_matrix(motif_match)

    return None


def match_motif_with_variants(
    mdata: MuData,
    motifs: Iterable,
    atac_mod: str = "atac",
    pseudocounts: float = 0.0001,
    p_value: float = 5e-05,
    background: _BACKGROUND = "even",
    key_added: str = "motif_match",
) -> None:

    return None


def tf_to_gene(
    mdata: MuData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    key_added: str = "gene_tf",
    inplace: bool = True,
) -> None | pd.DataFrame:
    """
    Link transcription factors (TFs) to target genes based on TF binding sites.

    This function identifies potential transcription factor regulators for each gene
    by mapping TF binding motifs in accessible chromatin regions (peaks) to target genes.
    It avoids self-regulation by removing cases where the target gene is also a TF.

    Parameters
    ----------
    mdata :
        A MuData object containing RNA and ATAC modalities.
    rna_mod:
        The name of the RNA modality in `mdata`.
    atac_mod:
        The name of the ATAC modality in `mdata`.
    key_added :
        The key under which the resulting dataframe is added to `mdata[rna_mod].uns`.
    inplace:
        If True, the results are added to `mdata`.
        If False, the resulting dataframe is returned.

    Returns
    -------
        If `inplace` is True, the function modifies `mdata` in place and returns None.
        If `inplace` is False, it returns a dataframe linking genes to TFs.

    Notes
    -----
        - Ensure that the peak-to-gene mapping (`peak_to_gene`) has been generated using `cell2net.pp.peak_to_gene` before running this function.
        - The input `mdata` must contain motif match information in `adata_atac.varm["motif_match"]`.

    Examples
    --------
    >>> import cell2net as cn
    >>> mdata = cn.example_data()
    >>> tf_to_gene(mdata)
    >>> result_df = tf_to_gene(mdata, inplace=False)
    >>> print(result_df)
    """
    adata_rna = mdata[rna_mod]
    adata_atac = mdata[atac_mod]

    assert (
        "peak_to_gene" in mdata.uns
    ), "Cannot find peak-to-gene, please first run cell2net.pp.peak_to_gene"

    genes = mdata.uns["peak_to_gene"]["gene"].unique().tolist()

    df_motifs = mdata.uns["motifs"]

    n_motifs = adata_atac.varm["motif_match"].shape[1]
    gene_tf = np.zeros(shape=(len(genes), n_motifs), dtype=np.uint8)

    logger.info("Find potential TFs for each gene")
    motif_names = df_motifs["gene_name"].values.tolist()
    for i, gene in enumerate(tqdm(genes)):
        df_p2g = mdata.uns["peak_to_gene"][mdata.uns["peak_to_gene"]["gene"] == gene]
        peaks = df_p2g["peak"].values.tolist()

        adata_atac_ = adata_atac[:, peaks]

        # for each peak, check which TFs are binding
        idx_list = []
        for j in range(len(peaks)):
            idx = adata_atac_.varm["motif_match"][j].nonzero()[1]  # type: ignore
            idx_list.append(idx)

        idx_list = list(set(np.concatenate(idx_list)))
        gene_tf[i, idx_list] = 1

        # to avoid self-regulation, we here check if the target gene is also a TF.
        # If so, remove it from the regulator
        if gene in motif_names:
            idx = motif_names.index(gene)
            gene_tf[i, idx] = 0

    df = pd.DataFrame(data=gene_tf, index=genes, columns=df_motifs["gene_name"])

    if inplace:
        adata_rna.uns[key_added] = df
    else:
        return df
