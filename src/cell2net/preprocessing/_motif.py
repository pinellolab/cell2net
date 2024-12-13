from collections.abc import Iterable
from typing import Literal, get_args

import MOODS.scan
import MOODS.tools
import numpy as np
import pandas as pd
from mudata import MuData
from pyjaspar import jaspardb
from scipy.sparse import csr_matrix
from tqdm import tqdm

from cell2net._logging import logger

_BACKGROUND = Literal["subject", "genome", "even"]


def get_motifs_from_jaspar(
    release: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: list[str] | None = None,
):
    """
    Fetch motifs from JASPAR database

    Parameters
    ----------
    release : str, optional
        _description_, by default "JASPAR2024"
    collection : str, optional
        _description_, by default "CORE"
    tax_group : list[str] | None, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if tax_group is None:
        tax_group = ["vertebrates"]

    jdb_obj = jaspardb(release=release)
    motifs = jdb_obj.fetch_motifs(collection=collection, tax_group=tax_group)

    return motifs


def match_motif(
    mdata: MuData,
    motifs: Iterable,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    pseudocounts=0.0001,
    p_value=5e-05,
    background: _BACKGROUND = "even",
    key_added: str = "motif_match",
) -> None:
    """
    Perform motif matching to predict binding sites using MOODS

    Parameters
    ----------
    mdata : MuData
        MuData object with RNA and ATAC modality.
    motifs : _type_
        List of motifs
    rna_mod: str, optional
        Name of RNA modality in mdata. Default: "rna"
    atac_mod: str, optional
        Name of ATAC modality in mdata. Default: "atac"
    pseudocounts : float, optional
        Pseudocounts for each nucleotide when performing motif matching. Default 0.0001
    p_value : _type_, optional
        P-value cut-off used to determine significant bidning sites. Default: 5e-05
    background : _BACKGROUND, optional
        Background distribution of nucleotides for computing thresholds from p-value.
        Three options are available: "subject" to use the subject sequences, "genome" to use the
        whole genome (need to provide a genome file), or even using 0.25 for each base,
        by default "even"
    genome_file : str, optional
        If background is set to genome, a genome file must be provided, by default None

    Returns
    -------
    Update mdata.
    """
    adata_rna = mdata[rna_mod]
    adata_atac = mdata[atac_mod]

    assert (
        "dna_sequence" in adata_atac.var.columns
    ), "Cannot find sequences, please first run cell2net.pp.add_dna_sequence"

    # overlap motifs and genes
    logger.info("Overlap motifs and genes")
    motif_names, motif_ids = [], []
    for motif in motifs:
        motif_names.append(motif.name)
        motif_ids.append(motif.matrix_id)

    df_motif = pd.DataFrame(data={"motif_name": motif_names, "motif_id": motif_ids})
    df_motif.drop_duplicates(subset=["motif_name"], keep="last", inplace=True)

    df_gene = pd.DataFrame(data={"gene_name": adata_rna.var_names})
    df_gene["gene_name_upper"] = df_gene["gene_name"].str.upper()

    sel_genes = list(
        set(df_gene["gene_name_upper"].values.tolist())
        & set(df_motif["motif_name"].values.tolist())
    )
    df_motif = df_motif[df_motif["motif_name"].isin(sel_genes)].reset_index(drop=True)
    df_gene = df_gene[df_gene["gene_name_upper"].isin(sel_genes)].reset_index(drop=True)

    assert len(df_motif) == len(df_gene), "Number of motifs and genes are different!"

    df_motif = pd.merge(
        df_motif, df_gene, left_on="motif_name", right_on="gene_name_upper", how="inner"
    )
    df_motif = df_motif[["motif_name", "motif_id", "gene_name"]]
    mdata.uns["motifs"] = df_motif

    # subset motifs
    motif_ids = df_motif["motif_id"].values.tolist()
    motifs_sub = []
    for motif in motifs:
        if motif.matrix_id in motif_ids:
            motifs_sub.append(motif)

    logger.info(f"Number of motifs overlapped with genes: {len(motifs_sub)}")

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
    mdata : MuData
        A MuData object containing RNA and ATAC modalities.
    rna_mod: str, optional
        The name of the RNA modality in `mdata`. Default is "rna".
    atac_mod: str, optional
        The name of the ATAC modality in `mdata`. Default is "atac".
    key_added : str, optional
        The key under which the resulting dataframe is added to `mdata[rna_mod].uns`.
        Default is "gene_tf".
    inplace: bool, optional
        If True, the results are added to `mdata`. If False, the resulting dataframe
        is returned. Default is True.

    Returns
    -------
    None or pd.DataFrame
        If `inplace` is True, the function modifies `mdata` in place and returns None.
        If `inplace` is False, it returns a dataframe linking genes to TFs.

    Examples
    --------
    >>> tf_to_gene(mdata)
    >>> result_df = tf_to_gene(mdata, inplace=False)

    Notes
    -----
    - Ensure that the peak-to-gene mapping (`peak_to_gene`) has been generated using
      `cell2net.pp.peak_to_gene` before running this function.
    - The input `mdata` must contain motif match information in `adata_atac.varm["motif_match"]`.
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
    for i, gene in tqdm(enumerate(genes)):
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
