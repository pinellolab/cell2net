import numpy as np
import pandas as pd
import vcfpy
from mudata import MuData
from tqdm import tqdm

from cell2net._logging import logger


def get_genomic_variants(reader: vcfpy.reader.Reader, chrom: str, start: int, end: int):
    """
    Extracts SNP (single nucleotide polymorphism) information and genotypes from a VCF file within a specified genomic region.

    Parameters
    ----------
    reader :
        A `vcfpy` VCF reader object initialized on the VCF file to be queried.
    chrom :
        Chromosome name to query (e.g., 'chr1' or '1').
    start :
        Start position of the genomic interval (1-based inclusive).
    end :
        End position of the genomic interval (1-based exclusive).

    Returns
    -------
        A DataFrame containing metadata for each SNP found in the region, including:
        - `id`: Variant ID from the VCF file.
        - `chrom`: Chromosome name.
        - `pos`: Genomic position (1-based).
        - `ref`: Reference allele.
        - `alt`: Alternate allele.
        - `sample`: Sample ID found in the VCF file.
        - `genotype`: Genotype of the sample for the SNP, encoded as:
            - 0 for homozygous reference (0/0)
            - 1 for heterozygous (0/1)
            - 2 for homozygous alternative (1/1)

    Raises
    ------
    AssertionError
        If a SNP record contains more than one alternative allele.

    Notes
    -----
        - Only SNVs (single nucleotide variants) are retained. Indels and multiallelic variants are skipped.
    """
    # Extract SNP information
    sample_ids = reader.header.samples.names  # type: ignore
    sample_ids = [str(x) for x in sample_ids]  # Ensure all elements are strings

    records = reader.fetch(chrom, start, end)

    snp_ids, snp_chroms, snp_positions, snp_refs, snp_alts = [], [], [], [], []
    genotypes = []
    for record in records:
        if record is None or not record.is_snv():
            continue

        assert (
            len(record.ALT) == 1
        ), f"find multiple alternatives for a SNP {record.ID[0]} "

        snp_ids.append(record.ID[0])
        snp_chroms.append(record.CHROM)
        snp_positions.append(record.POS)
        snp_refs.append(record.REF)
        snp_alts.append(record.ALT[0].value)

        # Extract genotype information
        genotype = [call.data.get("GT") or "./." for call in record.calls]
        genotype = [str(x) for x in genotype]  # Ensure all elements are strings
        genotype = [
            0 if x == "0/0" else 1 if x == "0/1" else 2 if x == "1/1" else np.nan
            for x in genotype
        ]

        genotypes.append(genotype)

    df_snp = pd.DataFrame(
        data={
            "chrom": snp_chroms,
            "pos": snp_positions,
            "ref": snp_refs,
            "alt": snp_alts,
        },
        index=snp_ids,
    )
    df_genotype = pd.DataFrame(
        data=genotypes, columns=sample_ids, index=snp_ids
    ).astype("Int8")

    df = pd.merge(df_snp, df_genotype, left_index=True, right_index=True)
    df["snp_id"] = df.index

    df = pd.melt(
        df,
        id_vars=["snp_id", "chrom", "pos", "ref", "alt"],
        var_name="sample",
        value_name="genotype",
    )
    df["snp_id"] = df["snp_id"].astype(str)
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(int)
    df["ref"] = df["ref"].astype(str)
    df["alt"] = df["alt"].astype(str)

    return df


def add_genomic_variants(
    mdata: MuData,
    vcf_file: str,
    atac_mod: str = "atac",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    variants_key: str = "variants",
    inpace: bool = True,
) -> None | pd.DataFrame:

    if atac_mod not in mdata.mod_names:
        logger.error(f"Cannot find modality: {atac_mod}")
        return None

    adata = mdata[atac_mod]

    reader = vcfpy.Reader.from_path(vcf_file)
    sample_ids = reader.header.samples.names  # type: ignore
    sample_ids = [str(x) for x in sample_ids]

    df_peaks = adata.var[[chr_var_key, start_var_key, end_var_key]]

    df_var_list = []
    for i, (chrom, start, end) in enumerate(
        tqdm(
            zip(
                df_peaks[chr_var_key],
                df_peaks[start_var_key],
                df_peaks[end_var_key],
                strict=False,
            ),
            total=len(df_peaks),
            desc="Processing variants",
        )
    ):
        df_var = get_genomic_variants(reader, chrom, start, end)
        df_var["peak"] = df_peaks.index[i]
        df_var_list.append(df_var)

    df_var = pd.concat(df_var_list, ignore_index=True)

    if inpace:
        adata.uns[variants_key] = df_var
        return None
    else:
        return df_var


# def add_genomic_variants(
#     mdata: MuData,
#     vcf_file: str | Path,
#     variants_key: str = "variants",
#     genotype_key: str = "genotype",
# ) -> None:
#     """
#     Adds genomic variant information from a VCF file to a MuData object.

#     This function reads single nucleotide variants (SNVs) from a VCF file and stores the
#     variant information and genotype data in the `uns` attribute of the MuData object.

#     Parameters
#     ----------
#     mdata :
#         A MuData object to which variant and genotype information will be added.
#     vcf_file :
#         Path to the VCF file containing genomic variant data
#     variants_key :
#         Key under which the variant information will be stored in `mdata.uns`.
#     genotype_key :
#         Key under which the genotype information will be stored in `mdata.uns`.

#     Notes
#     -----
#         - Only single nucleotide variants (SNVs) are considered.
#         - Variants with multiple alternative alleles are ignored.
#         - The extracted genotype data is encoded as:

#             - 0 for homozygous reference (0/0)
#             - 1 for heterozygous (0/1)
#             - 2 for homozygous alternative (1/1)

#         - The variant information is stored in `mdata.uns[variants_key]` as a pandas DataFrame with columns: `id`, `chrom`, `pos`, `ref`, and `alt`.
#         - The genotype information is stored in `mdata.uns[genotype_key]` as a pandas DataFrame where rows correspond to SNPs and columns correspond to samples.

#     Returns
#     -------
#         The function modifies `mdata` in place by adding variant and genotype information.

#     Raises
#     ------
#     AssertionError
#         If a SNP with multiple alternative alleles is encountered.

#     Examples
#     --------
#     >>> import muon as mu
#     >>> import cell2net as cn
#     >>> mdata = mu.MuData({})
#     >>> cn.pp.add_genomic_variants(mdata, "variants.vcf")
#     >>> mdata.uns["variants"].head()
#     >>> mdata.uns["genotype"].head()
#     """
#     reader = vcfpy.Reader.from_path(vcf_file)

#     sample_ids = reader.header.samples.names  # type: ignore
#     sample_ids = [str(x) for x in sample_ids]  # Ensure all elements are strings

#     snp_ids, snp_chroms, snp_positions, snp_refs, snp_alts = [], [], [], [], []
#     genotypes = []
#     for record in reader:
#         if record is None or not record.is_snv():
#             continue

#         assert (
#             len(record.ALT) == 1
#         ), f"find multiple alternatives for a SNP {record.ID[0]} "

#         snp_ids.append(record.ID[0])
#         snp_chroms.append(record.CHROM)
#         snp_positions.append(record.POS)
#         snp_refs.append(record.REF)
#         snp_alts.append(record.ALT[0].value)

#         # Extract genotype information
#         genotype = [call.data.get("GT") or "./." for call in record.calls]
#         genotype = [str(x) for x in genotype]  # Ensure all elements are strings
#         genotype = [
#             0 if x == "0/0" else 1 if x == "0/1" else 2 if x == "1/1" else np.nan
#             for x in genotype
#         ]

#         genotypes.append(genotype)

#     # Add SNP information to mdata
#     mdata.uns[variants_key] = pd.DataFrame(
#         data={
#             "id": snp_ids,
#             "chrom": snp_chroms,
#             "pos": snp_positions,
#             "ref": snp_refs,
#             "alt": snp_alts,
#         }
#     )

#     # Add donor information to mdata
#     mdata.uns[genotype_key] = pd.DataFrame(
#         data=genotypes, columns=sample_ids, index=snp_ids
#     ).astype("Int8")

#     return None


# def variant_to_peak(
#     mdata: MuData,
#     atac_mod: str = "atac",
#     variants_key: str = "variants",
#     peak_key: str = "peak",
# ) -> None:

#     # create ranges for peaks
#     pr_peaks = pr.PyRanges(
#         mdata[atac_mod].var[["chrom", "start", "end"]].rename(
#             columns={"start": "Start", "end": "End"}
#         )
#     )

#     df_peaks = mdata[atac_mod].var[["chrom", "start", "end"]].copy()

#     mdata["atac"].var["peak"] = (
#         mdata["atac"].var["chrom"]
#         + ":"
#         + mdata["atac"].var["start"].astype(str)
#         + "-"
#         + mdata["atac"].var["end"].astype(str)
#     )

#     pass
