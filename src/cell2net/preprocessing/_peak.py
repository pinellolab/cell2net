import logging
import re

import pandas as pd
import pyfaidx
import pyranges as pr
import pyranges.genomicfeatures as gf
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


def peak_to_gene(
    adata_rna: AnnData | MuData,
    adata_atac: AnnData | MuData,
    up_stream: int = 500_000,
    down_stream: int = 500_000,
    delimiter: str = "-",
    ref_fasta: str = "",
    use_veriable_gene=True,
):
    # Check if can find TSS coordinates in adata_rna
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
    adata_atac.var[["chrom", "start", "end"]] = adata_atac.var["peaks"].str.split(delimiter, expand=True)

    df_peaks = pd.DataFrame(
        data={
            "Chromosome": adata_atac.var["chrom"],
            "Start": adata_atac.var["start"],
            "End": adata_atac.var["end"],
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

    df = pd.concat(df_list)
    return df
