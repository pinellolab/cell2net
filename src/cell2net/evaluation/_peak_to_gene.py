import pandas as pd
import pyfaidx
import pyranges as pr
import pyranges.genomicfeatures as gf
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.preprocessing import get_gene_tss_coor

# def compute_enrichment(
#     pr_p2g: pr.PyRanges,
#     pr_genes: pr.PyRanges,
#     pr_causal_var: pr.PyRanges,
#     pr_common_var: pr.PyRanges,
# ) -> float:
#     # subset peak-to-gene links and gene coordiantes
#     pr_p2g_sub = pr_p2g[pr_p2g.Gene == gene]
#     pr_genes_sub = pr_genes[pr_genes.Gene == gene]
#     pr_causal_var_sub = pr_causal_var[pr_causal_var.Gene == gene]

#     # overlap causal variates with gene and peaks
#     pr_casual_var_in_peak = pr_causal_var_sub.overlap(pr_p2g_sub)
#     pr_casual_var_in_gene = pr_causal_var_sub.overlap(pr_genes_sub)

#     # overlap common variates with gene and peaks
#     pr_common_var_in_peak = pr_common_var.overlap(pr_p2g_sub)
#     pr_common_var_in_gene = pr_common_var.overlap(pr_genes_sub)

#     n_causal_var_in_peak = len(pr_casual_var_in_peak)
#     n_common_var_in_peak = len(pr_common_var_in_peak)
#     n_causal_var_in_gene = len(pr_casual_var_in_gene)
#     n_common_var_in_gene = len(pr_common_var_in_gene)

#     # compute enrichment
#     if n_common_var_in_peak == 0 or n_common_var_in_gene == 0:
#         enrichment = 0.0
#     elif n_causal_var_in_gene == 0:
#         enrichment = 0.0
#     else:
#         numerator = n_causal_var_in_peak / n_common_var_in_peak
#         denominator = n_causal_var_in_gene / n_common_var_in_gene
#         enrichment = numerator / denominator

#     return enrichment


def causal_var_enrichment_in_peaks(
    peak_to_gene: pd.DataFrame,
    causal_var: pd.DataFrame,
    common_var: pd.DataFrame,
    gene_gtf: str,
    ref_fasta: str,
    up_stream: int = 500_000,
    down_stream: int = 500_000,
    method_name: str = "Method",
) -> pd.DataFrame:
    pr_p2g = pr.PyRanges(peak_to_gene)

    # read causal and common variates
    pr_causal_var = pr.PyRanges(causal_var)
    pr_common_var = pr.PyRanges(common_var)

    # get tss for each gene
    logger.info("Fetch tss coordinates to generate background peaks")
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
    genes = peak_to_gene["Gene"].unique().tolist()
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
    df["method"] = method_name

    return df
