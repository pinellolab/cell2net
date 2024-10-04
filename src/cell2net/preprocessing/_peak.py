import logging
import re

import numpy as np
import pandas as pd
import pyfaidx
import pyranges as pr
import pyranges.genomicfeatures as gf
from mudata import MuData
from pysam import FastaFile
from tqdm import tqdm


def _seq_to_code(seq):
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(
            f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}"
        )

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }

    # Create array from nucleotide sequence
    vec = np.array([nuc_d[x] for x in seq], dtype=np.int8)

    return vec


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
    Add peak information to adata.var

    Parameters
    ----------
    mdata : MuData
        MuData object
    mod_name : str, optional
        Modality name, by default "atac"
    delimiter : str, optional
        Delimiter used to split the adata.var_names, by default "-"
    peak_len : int, optional
        Length of peaks, by default 256
    chr_var_key : str, optional
        _description_, by default "chr"
    start_var_key : str, optional
        _description_, by default "start"
    end_var_key : str, optional
        _description_, by default "end"

    Returns
    -------
    Update `mdata`
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


def add_dna_sequence_v2(
    mdata: MuData,
    ref_fasta: str,
    mod_name: str = "atac",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    obsm_key: str = "dna_one_hot",
):

    assert mod_name in mdata.mod_names, f"Cannot find modality: {mod_name}"
    adata = mdata[mod_name]

    fasta = FastaFile(filename=ref_fasta)
    df = adata.var[[chr_var_key, start_var_key, end_var_key]]

    data = np.empty(shape=(adata.n_obs, adata.n_vars, 4, 256), dtype=np.int8)

    # Loop for each chromosome
    for i, (chrom, start, end) in enumerate(
        zip(
            df[chr_var_key],
            df[start_var_key],
            df[end_var_key],
            strict=False,
        )
    ):

        seq = fasta.fetch(chrom, start, end).upper()
        data[:, i] = _seq_to_code(seq=seq)

    adata.obsm[obsm_key] = data

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
    """Add the DNA sequence of each peak to data object.

    Parameters
    ----------
    mdata : MuData
        MuData object
    ref_fasta : str
        Filename of genome reference
    delimiter : str, optional
        Delimiter that separates peaks, by default "-"

    Returns
    -------
    Update `mdata`
    """
    assert mod_name in mdata.mod_names, f"Cannot find modality: {mod_name}"
    adata = mdata[mod_name]

    fasta = FastaFile(filename=ref_fasta)
    df = adata.var[[chr_var_key, start_var_key, end_var_key]]

    seqs = []
    # Loop for each chromosome
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


def add_peak_to_gene(
    mdata: MuData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    up_stream: int = 500_000,
    down_stream: int = 500_000,
    ref_fasta: str = "",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
):
    # Check if can find TSS coordinates in adata_rna
    adata_rna = mdata[rna_mod]
    adata_atac = mdata[atac_mod]

    assert "gene_tss_coord" in adata_rna.uns, "Cannot find gene TSS coordinates"

    logging.info("Fetch gene coordinates")
    df_tss = adata_rna.uns["gene_tss_coord"]

    df_tss = adata_rna.uns["gene_tss_coord"]
    df_tss["Start"] = df_tss["tss"] - 1
    df_tss["End"] = df_tss["tss"]
    df_tss["Score"] = 0
    df_tss["Name"] = df_tss["gene_name"]
    df_tss["Strand"] = df_tss["strand"]
    df_tss["Chromosome"] = df_tss["chrom"]
    df_tss = df_tss[["Chromosome", "Start", "End", "Name", "Score", "Strand", "tss"]]

    gr_genes = pr.from_dict(df_tss)
    gr_genes = gr_genes.extend({"5": up_stream})
    gr_genes = gr_genes.extend({"3": down_stream})

    pyf = pyfaidx.Fasta(ref_fasta)
    gr_genes = gf.genome_bounds(gr_genes, chromsizes=pyf, clip=True)

    logging.info("Overlaping peaks with genes")
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

    mdata.uns["peak_to_gene"] = pd.concat(df_list).reset_index(drop=True)

    return None
