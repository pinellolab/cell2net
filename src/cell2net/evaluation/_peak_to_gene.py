import pandas as pd
import pyfaidx
import pyranges as pr
import pyranges.genomicfeatures as gf
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.preprocessing import get_gene_tss_coor


def causal_var_enrichment_in_peaks(
    df_p2g: pd.DataFrame,
    causal_var: pd.DataFrame,
    common_var: pd.DataFrame,
    gene_gtf: str,
    ref_fasta: str,
    up_stream: int = 500_000,
    down_stream: int = 500_000,
) -> pd.DataFrame:
    """
    Compute the enrichment of causal variants in peak regions linked to genes.

    This function calculates the enrichment of causal variants in regulatory peaks
    associated with genes. It compares the proportion of causal variants overlapping
    with peaks to the proportion of common variants overlapping with peaks,
    within the genomic window around the transcription start site (TSS).

    Parameters
    ----------
    df_p2g :
        DataFrame containing peak-to-gene (P2G) link information.
        Expected columns: ['peak', 'gene'].
        The `peak` column should be formatted as 'chrom-start-end'.
    causal_var :
        DataFrame containing causal variant information with
        columns ['Chromosome', 'Start', 'End', 'Gene'].
        Each row presents a link between causal variant and its target gene
    common_var :
        DataFrame containing common variant information with
        columns ['Chromosome', 'Start', 'End'].
    gene_gtf :
        File path to the gene annotation GTF file.
        This will be used to extract TSS for each gene.
    ref_fasta :
        File path to the reference genome FASTA file
    up_stream :
        Number of base pairs upstream of the TSS to include in the window, by default 500,000.
    down_stream : int, optional
        Number of base pairs downstream of the TSS to include in the window, by default 500,000

    Returns
    -------
        A DataFrame summarizing the enrichment results for each gene with following columns:
            - `gene`: Gene name.
            - `n_causal_var_in_peak`: Number of causal variants overlapping peaks.
            - `n_causal_var_in_gene`: Number of causal variants within the TSS window.
            - `n_common_var_in_peak`: Number of common variants overlapping peaks.
            - `n_common_var_in_gene`: Number of common variants within the TSS window.
            - `enrichment`: Enrichment score (ratio of causal-to-common variants in peaks vs. TSS window).

    Notes
    -----
        - The function utilizes PyRanges for efficient genomic range operations.
        - Variants not overlapping the TSS window or peaks are excluded from the enrichment calculation.
        - If no common variants are found within a gene's TSS window, that gene is skipped.
        - Enrichment is computed as:
            `enrichment = (n_causal_var_in_peak / n_causal_var_in_gene) / (n_common_var_in_peak / n_common_var_in_gene)`,
            with a default of 0 when no common variants overlap peaks or causal variants overlap the TSS window.

    """
    # Create PyRanges object for peak-to-gene links
    df_p2g[["Chromosome", "Start", "End"]] = df_p2g["peak"].str.split("-", expand=True)
    df_p2g = df_p2g[["Chromosome", "Start", "End", "gene"]]
    df_p2g = df_p2g.rename(columns={"gene": "Gene"})

    pr_p2g = pr.PyRanges(df_p2g)

    # read causal and common variates
    pr_causal_var = pr.PyRanges(causal_var)
    pr_common_var = pr.PyRanges(common_var)

    # get tss for each gene
    logger.info("Fetch tss coordinates to generate background regions")
    df_tss = get_gene_tss_coor(gene_gtf=gene_gtf, feature_type="gene")
    df_tss["Start"] = df_tss["tss"] - 1
    df_tss["End"] = df_tss["tss"]
    df_tss["Score"] = 0

    df_tss = df_tss.rename(
        columns={"chrom": "Chromosome", "strand": "Strand", "gene_name": "Gene"}
    )
    df_tss = df_tss[["Chromosome", "Start", "End", "Gene", "Score", "Strand", "tss"]]

    # convert to PyRanges object and extend by 500kb for each direction
    pr_tss = pr.PyRanges(df_tss)
    pr_tss = pr_tss.extend({"5": up_stream})
    pr_tss = pr_tss.extend({"3": down_stream})

    pyf = pyfaidx.Fasta(ref_fasta)
    pr_tss = gf.genome_bounds(pr_tss, chromsizes=pyf, clip=True)

    logger.info("Compute enrichment")
    genes = df_p2g["Gene"].unique().tolist()
    enrichment_list, gene_list = [], []
    n_causal_var_in_peak_list = []
    n_common_var_in_peak_list = []
    n_causal_var_in_gene_list = []
    n_common_var_in_gene_list = []
    for gene in tqdm(genes):
        # subset peak-to-gene links and gene coordiantes
        pr_p2g_sub = pr_p2g[pr_p2g.Gene == gene]
        pr_tss_sub = pr_tss[pr_tss.Gene == gene]
        pr_causal_var_sub = pr_causal_var[pr_causal_var.Gene == gene]

        # overlap causal variates with gene and peaks
        pr_casual_var_in_peak = pr_causal_var_sub.overlap(pr_p2g_sub)
        pr_casual_var_in_gene = pr_causal_var_sub.overlap(pr_tss_sub)

        # overlap common variates with gene and peaks
        pr_common_var_in_peak = pr_common_var.overlap(pr_p2g_sub)
        pr_common_var_in_gene = pr_common_var.overlap(pr_tss_sub)

        n_causal_var_in_peak = len(pr_casual_var_in_peak)
        n_common_var_in_peak = len(pr_common_var_in_peak)
        n_causal_var_in_gene = len(pr_casual_var_in_gene)
        n_common_var_in_gene = len(pr_common_var_in_gene)

        if n_common_var_in_gene == 0:
            continue

        if n_common_var_in_peak == 0 or n_causal_var_in_gene == 0:
            enrichment = 0
        else:
            numerator = n_causal_var_in_peak / n_causal_var_in_gene
            denominator = n_common_var_in_peak / n_common_var_in_gene
            enrichment = numerator / denominator

        gene_list.append(gene)
        enrichment_list.append(enrichment)
        n_causal_var_in_peak_list.append(n_causal_var_in_peak)
        n_common_var_in_peak_list.append(n_common_var_in_peak)
        n_causal_var_in_gene_list.append(n_causal_var_in_gene)
        n_common_var_in_gene_list.append(n_common_var_in_gene)

    df = pd.DataFrame(
        data={
            "gene": gene_list,
            "n_causal_var_in_peak": n_causal_var_in_peak_list,
            "n_causal_var_in_gene": n_causal_var_in_gene_list,
            "n_common_var_in_peak": n_common_var_in_peak_list,
            "n_common_var_in_gene": n_common_var_in_gene_list,
            "enrichment": enrichment_list,
        }
    )

    return df
