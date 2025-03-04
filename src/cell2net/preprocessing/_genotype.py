import re

import pandas as pd
from mudata import MuData
from pysam import FastaFile


def add_genotype(mdata: MuData, vcf_file: str) -> None:
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
