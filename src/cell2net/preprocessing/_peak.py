import logging
import re

import pandas as pd
import pyfaidx
import pyranges as pr
import pyranges.genomicfeatures as gf
from mudata import MuData
from pysam import FastaFile
from tqdm import tqdm


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


def peak_to_gene(
    mdata: MuData,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    up_stream: int = 500_000,
    down_stream: int = 500_000,
    ref_fasta: str = "",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    highly_variable: bool = True,
    genes: list[str] | None = None,
    min_n_peaks: int = 1,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    For each gene, identify its associated peaks limited by up and downstream.

    Parameters
    ----------
    mdata : MuData
        Input data object, should at least containing two modalities
    rna_mod : str, optional
        Name of RNA modality in mdata, by default "rna"
    atac_mod : str, optional
        Name of ATAC modality in mdata, by default "atac"
    up_stream : int, optional
        Distance of upstream of TSS to find associated peaks, by default 500_000
    down_stream : int, optional
        Distance of downstream of TSS to find associated peaks, by default 500_000
    ref_fasta : str, optional
        _description_, by default ""
    chr_var_key : str, optional
        Column name in mdata[atac_mod].var to get the chromosome of peaks, by default "chr"
    start_var_key : str, optional
        Column name in mdata[atac_mod].var to get the start position of peaks, by default "start"
    end_var_key : str, optional
        Column name in mdata[atac_mod].var to get the end position of peaks, by default "end"
    highly_variable : bool, optional
        Whether or not to only use highly variable genes, by default True
    genes : list[str] | None, optional
        Filter peak-to-gene list using these genes, by default None. If None, no filtering is performed
    min_n_peaks: int, optional
        Minimum number of associated peaks. Default: 1
    inplace: bool, optional
        If set, add the results to mdata, otherwise return the dataframe. Default: True

    Returns
    -------
    _type_
        Update mdata
    """
    # Check if can find TSS coordinates in adata_rna
    adata_rna = mdata[rna_mod]
    adata_atac = mdata[atac_mod]

    assert "gene_tss_coord" in adata_rna.uns, "Cannot find gene TSS coordinates"

    logging.info("Fetch gene coordinates")
    df_tss = adata_rna.uns["gene_tss_coord"]
    df_tss["Start"] = df_tss["tss"] - 1
    df_tss["End"] = df_tss["tss"]
    df_tss["Score"] = 0
    df_tss["Name"] = df_tss["gene_name"]
    df_tss["Strand"] = df_tss["strand"]
    df_tss["Chromosome"] = df_tss["chrom"]
    df_tss = df_tss[["Chromosome", "Start", "End", "Name", "Score", "Strand", "tss"]]

    if highly_variable:
        logging.info("Using highly variable genes")
        df = adata_rna.var[adata_rna.var["highly_variable"]]
        df_tss = df_tss[df_tss["Name"].isin(df["genes"])]

    if genes is not None:
        df_tss = df_tss[df_tss["Name"].isin(genes)]

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

    df = pd.concat(df_list).reset_index(drop=True)

    # Remove genes with number of associated peaks less than min_n_peaks
    grouped_df = df.groupby("gene").count()
    grouped_df = grouped_df[grouped_df["peak"] > min_n_peaks]

    df = df[df["gene"].isin(grouped_df.index)]

    if inplace:
        mdata.uns["peak_to_gene"] = df
    else:
        return df
