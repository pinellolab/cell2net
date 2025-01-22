import pandas as pd
import pyfaidx
import pyranges as pr
import pyranges.genomicfeatures as gf
from tqdm import tqdm

from cell2net._logging import logger
from cell2net.preprocessing import get_gene_tss_coord


def causal_var_enrichment_in_peaks(
    df_p2g: pd.DataFrame,
    causal_var: pd.DataFrame,
    common_var: pd.DataFrame,
    gene_gtf: str,
    ref_fasta: str,
    up_stream: int = 500_000,
    down_stream: int = 500_000,
) -> pd.DataFrame:
    r"""
    Compute the enrichment of causal variants in peak regions linked to genes.

    For each gene, this function calculates the enrichment of causal variants in
    gene-associated peaks by comparing the proportion of causal variants overlapping
    with peaks to the proportion of common variants overlapping with peaks
    after accounting for background regions defined as a genomic window around the
    transcription start site (TSS) for this gene.

    It can be used to evaluate the peak-to-gene links predicted by different models,
    such as Cell2net, SCARLink and SCENT.

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
        File path to the gene annotation GTF file which will be used to extract TSS
        for each gene to create background regions
    ref_fasta :
        File path to the reference genome FASTA file
    up_stream :
        Number of base pairs upstream of the TSS to include in the background region, by default 500,000.
    down_stream :
        Number of base pairs downstream of the TSS to include in the background region, by default 500,000

    Returns
    -------
        A DataFrame summarizing the enrichment results for each gene with following columns:

            - `gene`: Gene name.
            - `n_causal_var_in_peak`: Number of causal variants overlapping peaks.
            - `n_causal_var_in_background`: Number of causal variants within the background regions.
            - `n_common_var_in_peak`: Number of common variants overlapping peaks.
            - `n_common_var_in_background`: Number of common variants within the background regions.
            - `enrichment`: Enrichment score (ratio of causal-to-common variants in peaks vs. background regions).

    Notes
    -----
        - The function utilizes PyRanges for efficient genomic range operations.
        - Variants not overlapping the background regions or peaks are excluded from the enrichment calculation.
        - If no common variants are found within a gene's background regions, that gene is skipped.
        - Enrichment is computed as follows:

            .. math::
            enrichment = \\frac{\\frac{n_{\\text{causal var in peak}}}{n_{\\text{causal var in background}}}}{\\frac{n_{\\text{common var in peak}}}{n_{\\text{common var in background}}}

        - When no common variants overlap peaks or causal variants overlap the background regions.

    Example
    -------
        >>> df_p2g = pd.DataFrame({
        ...     'peak': ['chr1-1000-2000', 'chr1-3000-4000'],
        ...     'gene': ['GeneA', 'GeneB']
        ... })
        >>> causal_var = pd.DataFrame({
        ...     'Chromosome': ['chr1', 'chr1'],
        ...     'Start': [1500, 3500],
        ...     'End': [1501, 3501],
        ...     'Gene': ['GeneA', 'GeneB']
        ... })
        >>> common_var = pd.DataFrame({
        ...     'Chromosome': ['chr1', 'chr1'],
        ...     'Start': [1600, 3600],
        ...     'End': [1601, 3601]
        ... })
        >>> gene_gtf = "path/to/annotation.gtf"
        >>> ref_fasta = "path/to/reference.fasta"
        >>> result = causal_var_enrichment_in_peaks(
        ...     df_p2g, causal_var, common_var, gene_gtf, ref_fasta
        ... )
        >>> print(result)
            gene  n_causal_var_in_peak  n_causal_var_in_background  n_common_var_in_peak  n_common_var_in_background  enrichment
        0    GeneA                     1                    1                     1                    1    1.000000
        1    GeneB                     1                    1                     1                    1    1.000000
    """
    # create PyRanges object for peak-to-gene links
    _df_p2g = df_p2g.copy()

    _df_p2g[["Chromosome", "Start", "End"]] = _df_p2g["peak"].str.split(
        "-", expand=True
    )
    _df_p2g = _df_p2g[["Chromosome", "Start", "End", "gene"]]
    _df_p2g = _df_p2g.rename(columns={"gene": "Gene"})

    pr_p2g = pr.PyRanges(_df_p2g)

    # read causal and common variates
    pr_causal_var = pr.PyRanges(causal_var)
    pr_common_var = pr.PyRanges(common_var)

    # get tss for each gene
    logger.info("Get TSS coordinates to create background regions")
    df_tss = get_gene_tss_coord(gene_gtf=gene_gtf, feature_type="gene")
    df_tss["Start"] = df_tss["tss"] - 1
    df_tss["End"] = df_tss["tss"]
    df_tss["Score"] = 0

    df_tss = df_tss.rename(
        columns={"chrom": "Chromosome", "strand": "Strand", "gene_name": "Gene"}
    )
    df_tss = df_tss[["Chromosome", "Start", "End", "Gene", "Score", "Strand", "tss"]]

    # convert to PyRanges object and extend by 500kb for each direction
    pr_background = pr.PyRanges(df_tss)
    pr_background = pr_background.extend({"5": up_stream})
    pr_background = pr_background.extend({"3": down_stream})

    pyf = pyfaidx.Fasta(ref_fasta)
    pr_background = gf.genome_bounds(pr_background, chromsizes=pyf, clip=True)

    logger.info("Compute causal variants enrichment")
    genes = _df_p2g["Gene"].unique().tolist()
    n_peaks_list = []
    enrichment_list, gene_list = [], []
    n_causal_var_in_peak_list = []
    n_common_var_in_peak_list = []
    n_causal_var_in_background_list = []
    n_common_var_in_background_list = []
    for gene in tqdm(genes):
        # subset peak-to-gene links and background regions
        pr_p2g_sub = pr_p2g[pr_p2g.Gene == gene]
        pr_background_sub = pr_background[pr_background.Gene == gene]
        pr_causal_var_sub = pr_causal_var[pr_causal_var.Gene == gene]

        # overlap causal variates with peaks and background regions
        pr_casual_var_in_peak = pr_causal_var_sub.overlap(pr_p2g_sub)
        pr_casual_var_in_background = pr_causal_var_sub.overlap(pr_background_sub)

        # overlap common variates with peaks and background regions
        pr_common_var_in_peak = pr_common_var.overlap(pr_p2g_sub)
        pr_common_var_in_background = pr_common_var.overlap(pr_background_sub)

        n_causal_var_in_peak = len(pr_casual_var_in_peak)
        n_common_var_in_peak = len(pr_common_var_in_peak)
        n_causal_var_in_background = len(pr_casual_var_in_background)
        n_common_var_in_background = len(pr_common_var_in_background)

        if n_common_var_in_background == 0:
            continue

        if n_common_var_in_peak == 0 or n_causal_var_in_background == 0:
            enrichment = 0
        else:
            numerator = n_causal_var_in_peak / n_causal_var_in_background
            denominator = n_common_var_in_peak / n_common_var_in_background
            enrichment = numerator / denominator

        gene_list.append(gene)
        n_peaks_list.append(len(pr_p2g_sub))
        enrichment_list.append(enrichment)
        n_causal_var_in_peak_list.append(n_causal_var_in_peak)
        n_common_var_in_peak_list.append(n_common_var_in_peak)
        n_causal_var_in_background_list.append(n_causal_var_in_background)
        n_common_var_in_background_list.append(n_common_var_in_background)

    df = pd.DataFrame(
        data={
            "gene": gene_list,
            "n_peaks": n_peaks_list,
            "n_causal_var_in_peak": n_causal_var_in_peak_list,
            "n_causal_var_in_background": n_causal_var_in_background_list,
            "n_common_var_in_peak": n_common_var_in_peak_list,
            "n_common_var_in_background": n_common_var_in_background_list,
            "enrichment": enrichment_list,
        }
    )

    return df
