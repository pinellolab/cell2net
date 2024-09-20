import re

import pysam
from anndata import AnnData
from mudata import MuData


def add_peak_seq(data: AnnData | MuData, genome_file: str, delimiter="-"):
    """Add the DNA sequence of each peak to data object.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.
    genome_file : str
        Filename of genome reference
    delimiter : str, optional
        Delimiter that separates peaks, by default "-"

    Returns
    -------
    Update `data`
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")

    fasta = pysam.Fastafile(genome_file)
    adata.uns["peak_seq"] = [None] * adata.n_vars

    for i in range(adata.n_vars):
        peak = re.split(delimiter, adata.var_names[i])
        chrom, start, end = peak[0], int(peak[1]), int(peak[2])
        adata.uns["peak_seq"][i] = fasta.fetch(chrom, start, end).upper()

    return None
