import re

import pandas as pd
from mudata import MuData
from pysam import FastaFile


def add_genotype(mdata: MuData, ref_fasta: str, delimiter="-", mod_name: str = "atac"):
    """Add the DNA sequence of each peak to data object.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.
    ref_fasta : str
        Filename of genome reference
    delimiter : str, optional
        Delimiter that separates peaks, by default "-"

    Returns
    -------
    Update `mdata` by adding a modality containing the genotype information of each cell
    """
    assert mod_name in mdata.mod_names, f"Cannot find modality: {mod_name}"
    adata = mdata[mod_name]

    fasta = FastaFile(filename=ref_fasta)
    peaks, seqs = [], []
    for i in range(adata.n_vars):
        peak = re.split(delimiter, adata.var_names[i])
        chrom, start, end = peak[0], int(peak[1]), int(peak[2])
        peaks.append(adata.var_names[i])
        seqs.append(fasta.fetch(chrom, start, end).upper())

    adata.uns["peak_seq"] = pd.DataFrame(data={"peak": peaks, "seq": seqs})
    return None
