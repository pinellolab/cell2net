import pandas as pd
import pyfaidx
import pyranges as pr
import pyranges.genomicfeatures as gf
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.preprocessing import get_gene_tss_coor


def compute_enrichment(
    pr_p2g: pr.PyRanges,
    pr_genes: pr.PyRanges,
    gene: str,
    pr_causal_var: pr.PyRanges,
    pr_common_var: pr.PyRanges,
) -> float:
    # subset peak-to-gene links and gene coordiantes
    pr_p2g_sub = pr_p2g[pr_p2g.Gene == gene]
    pr_genes_sub = pr_genes[pr_genes.Gene == gene]
    pr_causal_var_sub = pr_causal_var[pr_causal_var.Gene == gene]

    # overlap causal variates with gene and peaks
    pr_casual_var_in_peak = pr_causal_var_sub.overlap(pr_p2g_sub)
    pr_casual_var_in_gene = pr_causal_var_sub.overlap(pr_genes_sub)

    # overlap common variates with gene and peaks
    pr_common_var_in_peak = pr_common_var.overlap(pr_p2g_sub)
    pr_common_var_in_gene = pr_common_var.overlap(pr_genes_sub)

    n_causal_var_in_peak = len(pr_casual_var_in_peak)
    n_common_var_in_peak = len(pr_common_var_in_peak)
    n_causal_var_in_gene = len(pr_casual_var_in_gene)
    n_common_var_in_gene = len(pr_common_var_in_gene)

    # compute enrichment
    if n_common_var_in_peak == 0 or n_common_var_in_gene == 0:
        enrichment = 0.0
    elif n_causal_var_in_gene == 0:
        enrichment = 0.0
    else:
        numerator = n_causal_var_in_peak / n_common_var_in_peak
        denominator = n_causal_var_in_gene / n_common_var_in_gene
        enrichment = numerator / denominator

    return enrichment


def causal_var_enrichment_in_peaks(
    peak_to_gene: pd.DataFrame,
    causal_var: str,
    common_var: str,
    gene_gtf: str,
    ref_fasta: str,
) -> pd.DataFrame:
    """
    Compute enrichment of causal variantes in gene-associated peaks

    Parameters
    ----------
    df_p2g : pd.DataFrame
        A dataframe of peak-to-gene links.
        Must have the follow columns: "Chromosome", "Start", "End", "Gene"
    df_causal_var : str
        A dataframe of causal variants
        Must have the follow columns: "Chromosome", "Start", "End", "Gene"
    common_variants : str
        _description_
    gene_gtf : str
        _description_
    ref_fasta : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    pr_p2g = pr.PyRanges(peak_to_gene)

    # read causal variates
    pr_causal_var = pr.PyRanges(causal_var)
    pr_common_var = pr.PyRanges(common_var)

    # get tss and extend by 500kb
    logger.info("Load gene coordinates")
    df_genes = get_gene_tss_coor(gene_gtf=gene_gtf)

    df_genes["Start"] = df_genes["tss"] - 1
    df_genes["End"] = df_genes["tss"]
    df_genes["Score"] = 0

    df_genes = df_genes.rename(
        columns={"chrom": "Chromosome", "strand": "Strand", "gene_name": "Gene"}
    )
    df_genes = df_genes[
        ["Chromosome", "Start", "End", "Gene", "Score", "Strand", "tss"]
    ]

    pr_genes = pr.from_dict(df_genes)
    pr_genes = pr_genes.extend({"5": 500_000})
    pr_genes = pr_genes.extend({"3": 500_000})

    pyf = pyfaidx.Fasta(ref_fasta)
    pr_genes = gf.genome_bounds(pr_genes, chromsizes=pyf, clip=True)

    logger.info("Compute enrichment")
    genes = df_p2g["Gene"].unique().tolist()
    enrichment_list, gene_list = [], []
    for gene in tqdm(genes):
        # subset peak-to-gene links and gene coordiantes
        pr_p2g_sub = pr_p2g[pr_p2g.Gene == gene]
        pr_genes_sub = pr_genes[pr_genes.Gene == gene]
        pr_causal_var_sub = pr_causal_var[pr_causal_var.Gene == gene]

        # overlap causal variates with gene and peaks
        pr_casual_var_in_peak = pr_causal_var_sub.overlap(pr_p2g_sub)
        pr_casual_var_in_gene = pr_causal_var_sub.overlap(pr_genes_sub)

        # overlap common variates with gene and peaks
        pr_common_var_in_peak = pr_common_var.overlap(pr_p2g_sub)
        pr_common_var_in_gene = pr_common_var.overlap(pr_genes_sub)

        n_causal_var_in_peak = len(pr_casual_var_in_peak)
        n_common_var_in_peak = len(pr_common_var_in_peak)
        n_causal_var_in_gene = len(pr_casual_var_in_gene)
        n_common_var_in_gene = len(pr_common_var_in_gene)

        if n_common_var_in_gene == 0:
            continue

        if n_common_var_in_peak == 0 or n_causal_var_in_gene == 0:
            gene_list.append(gene)
            enrichment_list.append(0)
            continue

        numerator = n_causal_var_in_peak / n_common_var_in_peak
        denominator = n_causal_var_in_gene / n_common_var_in_gene
        enrichment = numerator / denominator

        enrichment_list.append(enrichment)
        gene_list.append(gene)

    df = pd.DataFrame(data={"gene": gene_list, "enrichment": enrichment_list})

    return df
