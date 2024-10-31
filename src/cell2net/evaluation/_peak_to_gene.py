import pandas as pd


def variant_enrichment(peak_to_gene: pd.DataFrame, causal_variants, common_variants):

    # Step 1a: Load GTEx data and filter for fine-mapped variants with PIP > 0.2
    gtex_file = "/data/pinello/PROJECTS/2023_09_JF_SIMBAvariant/benchmark/SCENT/GTEx_v8_finemapping_CAVIAR/CAVIAR_Results_v8_GTEx_LD_ALL_NOCUTOFF_with_Allele.txt.gz"
    gtex_df = pd.read_csv(gtex_file, compression="gzip", sep="\t")
    gtex_df["CHROM"] = gtex_df["CHROM"].astype(str)
    fine_mapped_variants = gtex_df[gtex_df["Probability"] > 0.2]

    return None
