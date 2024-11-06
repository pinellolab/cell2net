from typing import Literal, get_args

import MOODS.scan
import MOODS.tools
import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from pyjaspar import jaspardb
from tqdm import tqdm

_BACKGROUND = Literal["subject", "genome", "even"]


def get_motifs_from_jaspar(
    release: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: list[str] | None = None,
):

    if tax_group is None:
        tax_group = ["vertebrates"]

    jdb_obj = jaspardb(release=release)
    motifs = jdb_obj.fetch_motifs(collection=collection, tax_group=tax_group)

    return motifs


def add_tf_info_from_jaspar(
    mdata: MuData,
    rna_mod: str = "rna",
    motifs: list | None = None,
) -> None:
    """
    Check if the genes are transcription factor using JASPAR database.

    Parameters
    ----------
    mdata : MuData
        Input MuData object containing gene expression
    mod_names : str
        Name of RNA modality in mdata, by default "rna"
    release : str
        Release of JASPAR database, by default "JASPAR2024"
    """
    assert rna_mod in mdata.mod_names, f"Cannot find modality: {rna_mod}"
    adata = mdata[rna_mod]

    jdb_obj = jaspardb(release=release)
    motifs = jdb_obj.fetch_motifs(collection="CORE", tax_group=["vertebrates"])

    motif_names, motif_ids = [], []
    for motif in motifs:
        motif_names.append(motif.name)
        motif_ids.append(motif.matrix_id)

    df_motif = pd.DataFrame(data={"name": motif_names, "matrix_id": motif_ids})

    # tf_names = []
    # for motif in motifs:
    #     tf_names.append(motif.name)

    # adata.var_names.upper()

    is_tf = []
    for gene_name in adata.var_names:
        if gene_name in df_motif["name"] or gene_name.upper() in df_motif["name"]:
            is_tf.append(True)
        else:
            is_tf.append(False)

    adata.var["is_tf"] = is_tf

    return None


def match_motif(
    data: MuData | AnnData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    pseudocounts=0.0001,
    p_value=5e-05,
    background: _BACKGROUND = "even",
) -> None:
    """
    Perform motif matching to predict binding sites using MOODS

    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.
    motifs : _type_
        List of motifs
    pseudocounts : float, optional
        Pseudocounts for each nucleotide, by default 0.0001
    p_value : _type_, optional
        _description_, by default 5e-05
    background : _BACKGROUND, optional
        Background distribution of nucleotides for computing thresholds from p-value.
        Three options are available: "subject" to use the subject sequences, "genome" to use the
        whole genome (need to provide a genome file), or even using 0.25 for each base,
        by default "even"
    genome_file : str, optional
        If background is set to genome, a genome file must be provided, by default None

    Returns
    -------
    Update data.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and atac_mod in data.mod:
        adata = data.mod[atac_mod]
    else:
        raise TypeError(f"Expected AnnData or MuData object with {atac_mod} modality")

    assert (
        "dna_sequence" in adata.var.columns
    ), "Cannot find sequences, please first run cell2net.pp.add_dna_sequence"

    options = get_args(_BACKGROUND)
    assert background in options, f"'{background}' is not in {options}"

    jdb_obj = jaspardb(release="JASPAR2024")
    motifs = jdb_obj.fetch_motifs(collection="CORE", tax_group=["vertebrates"])

    # add motif names to Anndata object
    adata.uns["motif_name"] = [None] * len(motifs)
    for i, motif in enumerate(motifs):
        adata.uns["motif_name"][i] = motif.matrix_id + "." + motif.name

    # compute background distribution
    seq = ""
    if background == "subject":
        for i in range(adata.n_vars):
            seq += adata.uns["peak_seq"][i]
        _bg = MOODS.tools.bg_from_sequence_dna(seq, 0)
    elif background == "genome":
        # TODO
        _bg = MOODS.tools.flat_bg(4)
    else:
        _bg = MOODS.tools.flat_bg(4)

    # prepare motif data
    n_motifs = len(motifs)

    matrices = [None] * 2 * n_motifs
    thresholds = [None] * 2 * n_motifs
    for i, motif in enumerate(motifs):
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
    adata.varm["motif_match"] = np.zeros(shape=(adata.n_vars, n_motifs), dtype=np.uint8)

    for i in tqdm(range(adata.n_vars)):
        results = scanner.scan(adata.var["dna_sequence"].iloc[i])
        for j in range(n_motifs):
            if len(results[j]) > 0 or len(results[j + n_motifs]) > 0:
                adata.varm["motif_match"][i, j] = 1  # type: ignore

    return None
