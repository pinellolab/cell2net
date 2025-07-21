from collections.abc import Iterable

import MOODS.scan
import MOODS.tools
import numpy as np
import pandas as pd
from mudata import MuData
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from cell2net._logging import logger


def _get_motifs_from_file():
    """
    Load transcription factor motifs from a file.

    This function is a placeholder for loading motifs from a file.
    It should be implemented to read motif data from a specified file format.

    Returns
    -------
        An iterable of motif objects loaded from the file.
        Returns `None` if the file cannot be read or is not implemented.
    """
    logger.error("This function is not implemented yet.")
    return None

def _get_motifs_from_jaspar(
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


def get_tf_motifs(database: str) -> Iterable:
    """
    Fetch transcription factor motifs from a specified database.

    Parameters
    ----------
    database : str
        The name of the database to fetch motifs from. Currently, only "JASPAR2024" is supported.

    Returns
    -------
    Iterable | None
        A collection of transcription factor motifs or None if an error occurs.
    """
    if database == "JASPAR2024":
        return _get_motifs_from_jaspar(
            release="JASPAR2024", collection="CORE", tax_group=["vertebrates"]
        )
    else:
        logger.error(f"Database {database} is not supported.")
        return None


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


def prepare_scaner(
    motifs: list,
    pseudocounts: float = 0.0001,
    p_value: float = 5e-05,
) -> MOODS.scan.Scanner:
    """
    Prepares a MOODS scanner for motif scanning.

    This function initializes a `MOODS.scan.Scanner` object with the given motifs,
    applying log-odds transformation to the motif counts and computing motif-specific
    score thresholds based on a given p-value. It also generates reverse complement
    motifs for scanning both DNA strands.


    Parameters
    ----------
    motifs :
         A list of motif objects containing nucleotide frequency counts
        (expected to have `.counts` attribute with keys "A", "C", "G", and "T").
    pseudocounts :
        A small pseudocount added to avoid zero probabilities in log-odds calculation.
    p_value :
        The statistical significance threshold for computing motif scoring thresholds.

    Returns
    -------
        A `MOODS.scan.Scanner` object initialized with the provided motifs and
        scoring thresholds, ready for scanning DNA sequences.

    Notes
    -----
        - The function creates both **original** and **reverse complement** motifs for scanning.
        - The background nucleotide frequency is assumed to be uniform (flat background).
        - `MOODS.tools.threshold_from_p()` is used to compute the score thresholds.

    Examples
    --------
    >>> import cell2net as cn
    >>> motifs = [some_motif_object1, some_motif_object2]  # Assume motif objects are loaded
    >>> scanner = cn.pp.prepare_scaner(motifs)
    >>> type(scanner)
    <class 'MOODS.scan.Scanner'>
    """
    n_motifs = len(motifs)
    bg = MOODS.tools.flat_bg(4)

    matrices = [None] * 2 * n_motifs
    thresholds = [None] * 2 * n_motifs
    for i, motif in enumerate(motifs):
        counts = (
            tuple(motif.counts["A"]),
            tuple(motif.counts["C"]),
            tuple(motif.counts["G"]),
            tuple(motif.counts["T"]),
        )

        matrices[i] = MOODS.tools.log_odds(counts, bg, pseudocounts)
        matrices[i + n_motifs] = MOODS.tools.reverse_complement(matrices[i])

        thresholds[i] = MOODS.tools.threshold_from_p(matrices[i], bg, p_value)
        thresholds[i + n_motifs] = thresholds[i]

    scanner = MOODS.scan.Scanner(7)
    scanner.set_motifs(matrices=matrices, bg=bg, thresholds=thresholds)

    return scanner


def match_motif_with_seq(
    motifs: list, seq: list[str], pseudocounts: float = 0.0001, p_value: float = 5e-05
) -> np.ndarray:
    """
    Match a list of sequence motifs to a list of DNA sequences.

    This function scans each DNA sequence in the input list for occurrences of given motifs
    (including both forward and reverse complements), using a scanner initialized with specified
    pseudocounts and p-value threshold. It returns a binary matrix indicating which motifs are
    present in which sequences.

    Parameters
    ----------
    motifs :
        A list of motif objects (e.g., PWM or PSSM representations) to scan for.
    seq :
        A list of DNA sequences (as strings) to be scanned for motif matches.
    pseudocounts :
        Pseudocount value added to motif frequencies to avoid zero probabilities (default is 0.0001).
    p_value :
        P-value threshold for motif match significance (default is 5e-05).

    Returns
    -------
        A 2D NumPy array of shape (n_sequences, n_motifs) with binary values.
        Each entry [i, j] is 1 if motif `j` is found in sequence `i` (in either strand), and 0 otherwise.
    """
    n_motifs = len(motifs)
    motif_match = np.zeros(shape=(len(seq), len(motifs)), dtype=np.uint8)
    scanner = prepare_scaner(motifs=motifs, pseudocounts=pseudocounts, p_value=p_value)

    for i in range(len(seq)):
        results = scanner.scan(seq[i])
        for j in range(n_motifs):
            if len(results[j]) > 0 or len(results[j + n_motifs]) > 0:
                motif_match[i, j] = 1  # type: ignore

    return motif_match


def match_motif(
    mdata: MuData,
    motifs: Iterable,
    atac_mod: str = "atac",
    pseudocounts: float = 0.0001,
    p_value: float = 5e-05,
    sequence_var_key: str = "dna_sequence",
    key_added: str = "motif_match",
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

    key_added :
        Name of the key to store the resulting motif match matrix in `adata_atac.varm`.

    Returns
    -------
    Results are added to the `mdata` object in place:

        - `mdata[atac_mod].varm[key_added]`: Sparse matrix indicating motif matches for each accessible DNA sequence.

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
    if atac_mod not in mdata.mod:
        logger.error(f"Cannot find {atac_mod} in mdata, please check the name!")

    adata = mdata[atac_mod]

    if sequence_var_key not in mdata[atac_mod].var.columns:
        logger.error(
            "Cannot find sequences, please first run cell2net.pp.add_dna_sequence"
        )

    # get motifs
    motif_ids = mdata.uns["motifs"]["motif_id"].values.tolist()
    motifs_sub = []
    for motif in motifs:
        if motif.matrix_id in motif_ids:
            motifs_sub.append(motif)

    n_motifs = len(motifs_sub)
    logger.info(f"Number of motifs: {n_motifs}")

    logger.info("Matching TF motifs")
    motif_match = match_motif_with_seq(
        motifs=motifs_sub,
        seq=adata.var[sequence_var_key].values.tolist(),
        pseudocounts=pseudocounts,
        p_value=p_value,
    )

    adata.varm[key_added] = csr_matrix(motif_match)
    logger.info("Motif matching is done!")

    return None


def match_motif_with_variants(
    mdata: MuData,
    motifs: Iterable,
    atac_mod: str = "atac",
    pseudocounts: float = 0.0001,
    p_value: float = 5e-05,
    seq_with_variants_key: str = "seq_with_variants",
    n_cpus: int = 1,
) -> None:
    """
    Match transcription factor (TF) motifs to variant-aware genomic sequences across samples in a MuData object.

    This function scans reference and variant-altered sequences for motif matches across all peaks
    and samples in the specified modality of a `MuData` object. It stores the binary motif match
    results (presence/absence) for each peak and sample in `adata.varm`, optionally using
    parallel processing.

    Parameters
    ----------
    mdata :
        A MuData object containing multi-modal single-cell data, including an ATAC modality with
        per-sample variant-aware sequences stored in `.uns[seq_with_variants_key]`.
    motifs :
         A list or iterable of motif objects (e.g., from `MOODS` or `Bio.motifs`) to match against the sequences.
    atac_mod :
        Name of the modality in `mdata` that contains the ATAC-seq data.
    pseudocounts :
        Pseudocount value added to motif frequencies to avoid zero probabilities
    p_value :
        P-value threshold for motif match significance
    seq_with_variants_key :
        Key in `.uns` of `adata` corresponding to a DataFrame with variant-aware sequences (`seq_1`, `seq_2`)
        for each peak and sample
    n_cpus :
        Number of CPUs to use for parallel motif scanning. If `n_cpus > 1`, parallelization is enabled

    Returns
    -------
        The function updates the `.varm` attribute of the specified `adata` (i.e., `mdata[atac_mod]`) in-place.
        For each sample, a sparse binary matrix is stored, indicating motif presence/absence per peak.

    Notes
    -----
        - This function requires a `prepare_scaner` function and a compatible `match_motif_with_seq` function.
        - It assumes the presence of a DataFrame in `.uns[seq_with_variants_key]` with the following columns:
            - "sample": sample ID
            - "peak": peak ID (matching `adata.var.index`)
            - "seq_1": reference or original sequence
            - "seq_2": variant-altered sequence
        - The final result stores a binary sparse matrix in `.varm[sample_id]`, with rows as peaks and columns as motifs.

    Example
    -------
    >>> from mudata import MuData
    >>> from some_motif_library import load_motifs
    >>> mdata = MuData.read_h5mu("multiome_data.h5mu")
    >>> motifs = load_motifs("JASPAR2022_CORE.meme")
    >>> match_motif_with_variants(mdata, motifs, atac_mod="atac", n_cpus=4)
    """
    if atac_mod not in mdata.mod:
        logger.error(f"Cannot find {atac_mod} in mdata, please check the name!")

    adata = mdata[atac_mod]

    if seq_with_variants_key not in adata.uns:
        logger.error(
            f"Cannot find {seq_with_variants_key}, please first run cell2net.pp.add_variants_to_sequence"
        )

    df_seq = adata.uns[seq_with_variants_key]

    sample_list = df_seq["sample"].unique()

    # Get motifs
    motif_ids = mdata.uns["motifs"]["motif_id"].values.tolist()
    motifs_sub = []
    for motif in motifs:
        if motif.matrix_id in motif_ids:
            motifs_sub.append(motif)

    n_motifs = len(motifs_sub)
    logger.info(f"Number of motifs: {n_motifs}")

    logger.info("Matching TF motifs with variants information")

    if n_cpus > 1:
        logger.info(f"Using {n_cpus} CPUs for parallel processing.")
        from multiprocessing import Pool

        # split the sequences by sample
        # and run the match_motif_with_seq in parallel
        args_1, args_2 = [], []
        for sample_id in sample_list:
            _df_seq = df_seq[df_seq["sample"] == sample_id].set_index("peak")

            if len(_df_seq) != adata.n_vars:
                logger.error(
                    f"Sample {sample_id} does not have all peaks, please check the sample!"
                )
                continue

            _df_seq = _df_seq.loc[adata.var.index,]

            args_1.append(
                (motifs_sub, _df_seq["seq_1"].values.tolist(), pseudocounts, p_value)
            )
            args_2.append(
                (motifs_sub, _df_seq["seq_2"].values.tolist(), pseudocounts, p_value)
            )

        with Pool(n_cpus) as pool:
            results_1 = pool.starmap(match_motif_with_seq, args_1)
            results_2 = pool.starmap(match_motif_with_seq, args_2)

        for sample_id, result_1, result_2 in zip(
            sample_list, results_1, results_2, strict=False
        ):
            result = np.logical_or(result_1, result_2).astype(np.uint8)

            adata.varm[sample_id] = csr_matrix(result)

    else:
        # for each donor, scan the motifs across all peaks
        for sample_id in tqdm(
            sample_list, desc="Matching motifs", total=len(sample_list)
        ):

            # get the sequence for the sample
            _df_seq = df_seq[df_seq["sample"] == sample_id].set_index("peak")

            # make sure each sample has all peaks
            if len(_df_seq) != adata.n_vars:
                logger.error(
                    f"Sample {sample_id} does not have all peaks, please check the sample!"
                )
                continue

            # resort the dataframe
            _df_seq = _df_seq.loc[adata.var.index,]

            motif_match_1 = match_motif_with_seq(
                motifs=motifs_sub,
                seq=_df_seq["seq_1"].values.tolist(),
                pseudocounts=pseudocounts,
                p_value=p_value,
            )

            motif_match_2 = match_motif_with_seq(
                motifs=motifs_sub,
                seq=_df_seq["seq_2"].values.tolist(),
                pseudocounts=pseudocounts,
                p_value=p_value,
            )

            motif_match = np.logical_or(motif_match_1, motif_match_2).astype(np.uint8)
            adata.varm[sample_id] = csr_matrix(motif_match)

    logger.info("Motif matching is done!")

    return None


def tf_to_gene(
    mdata: MuData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    peak_to_gene_key: str = "peak_to_gene",
    motif_match_key: str = "motif_match",
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
    peak_to_gene_key :
        The key under which the peak-to-gene mapping is stored in `mdata.uns`.
    motif_match_key :
        The key under which the motif match information is stored in `mdata[atac_mod].varm`.
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
    if atac_mod not in mdata.mod:
        logger.error(f"Cannot find {atac_mod} in mdata, please check the name!")
    if rna_mod not in mdata.mod:
        logger.error(f"Cannot find {rna_mod} in mdata, please check the name!")

    adata_rna = mdata[rna_mod]
    adata_atac = mdata[atac_mod]

    if peak_to_gene_key not in mdata.uns:
        logger.error(
            "Cannot find peak-to-gene, please first run cell2net.pp.peak_to_gene"
        )
    if motif_match_key not in adata_atac.varm:
        logger.error(
            "Cannot find motif match, please first run cell2net.pp.match_motif"
        )

    genes = mdata.uns[peak_to_gene_key]["gene"].unique().tolist()
    df_motifs = mdata.uns["motifs"]
    n_motifs = adata_atac.varm[motif_match_key].shape[1]
    gene_tf = np.zeros(shape=(len(genes), n_motifs), dtype=np.uint8)

    logger.info("Find potential TFs for each gene")
    motif_names = df_motifs["gene_name"].values.tolist()
    for i, gene in enumerate(tqdm(genes, desc="Finding TFs", total=len(genes))):
        df_p2g = mdata.uns[peak_to_gene_key][
            mdata.uns[peak_to_gene_key]["gene"] == gene
        ]
        peaks = df_p2g["peak"].values.tolist()

        adata_atac_ = adata_atac[:, peaks]

        # for each peak, check which TFs are binding
        idx_list = []
        for j in range(len(peaks)):
            idx = adata_atac_.varm[motif_match_key][j].nonzero()[1]  # type: ignore
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


def tf_to_gene_with_variants(
    mdata: MuData,
    samples: list[str] | None = None,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    peak_to_gene_key: str = "peak_to_gene",
    key_added: str = "gene_tf",
) -> None | pd.DataFrame:

    if atac_mod not in mdata.mod:
        logger.error(f"Cannot find {atac_mod} in mdata, please check the name!")
    if rna_mod not in mdata.mod:
        logger.error(f"Cannot find {rna_mod} in mdata, please check the name!")

    adata_rna = mdata[rna_mod]
    adata_atac = mdata[atac_mod]

    if peak_to_gene_key not in mdata.uns:
        logger.error(
            "Cannot find peak-to-gene, please first run cell2net.pp.peak_to_gene"
        )

    genes = mdata.uns[peak_to_gene_key]["gene"].unique().tolist()
    df_motifs = mdata.uns["motifs"]
    # n_motifs = adata_atac.varm[motif_match_key].shape[1]

    motif_names = df_motifs["gene_name"].values.tolist()

    # link tf to genes for each samlpe based on the peak-to-gene mapping
    # and the motif match information
    # samples = list(set(samples) & set(adata_atac.varm.keys()))

    if samples is None:
        samples = list(adata_atac.varm.keys())

    df_p2g = mdata.uns[peak_to_gene_key]

    adata_rna.uns[key_added] = {}
    for sample in tqdm(samples, desc="Linking tf to genes", total=len(samples)):
        n_motifs = adata_atac.varm[sample].shape[1]

        gene_tf = np.zeros(shape=(len(genes), n_motifs), dtype=np.uint8)
        for i, gene in enumerate(genes):
            peaks = df_p2g[df_p2g["gene"] == gene]["peak"].values.tolist()

            adata_atac_ = adata_atac[:, peaks]
            # for each peak, check which TFs are binding
            idx_list = []
            for j in range(len(peaks)):
                idx = adata_atac_.varm[sample][j].nonzero()[1]  # type: ignore
                idx_list.append(idx)

            idx_list = list(set(np.concatenate(idx_list)))
            gene_tf[i, idx_list] = 1

            # to avoid self-regulation, we here check if the target gene is also a TF.
            # If so, remove it from the regulator
            if gene in motif_names:
                idx = motif_names.index(gene)
                gene_tf[i, idx] = 0

        df = pd.DataFrame(data=gene_tf, index=genes, columns=df_motifs["gene_name"])
        adata_rna.uns[key_added][sample] = df

    return None
