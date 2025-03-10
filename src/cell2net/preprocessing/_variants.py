from pathlib import Path

import numpy as np
import pandas as pd
import vcfpy
from mudata import MuData


def add_genomic_variants(
    mdata: MuData,
    vcf_file: str | Path,
    variants_key: str = "variants",
    genotype_key: str = "genotype",
) -> None:
    """
    Adds genomic variant information from a VCF file to a MuData object.

    This function reads single nucleotide variants (SNVs) from a VCF file and stores the
    variant information and genotype data in the `uns` attribute of the MuData object.

    Parameters
    ----------
    mdata :
        A MuData object to which variant and genotype information will be added.
    vcf_file :
        Path to the VCF file containing genomic variant data
    variants_key :
        Key under which the variant information will be stored in `mdata.uns`.
    genotype_key :
        Key under which the genotype information will be stored in `mdata.uns`.

    Notes
    -----
        - Only single nucleotide variants (SNVs) are considered.
        - Variants with multiple alternative alleles are ignored.
        - The extracted genotype data is encoded as:

            - 0 for homozygous reference (0/0)
            - 1 for heterozygous (0/1)
            - 2 for homozygous alternative (1/1)

        - The variant information is stored in `mdata.uns[variants_key]` as a pandas DataFrame with columns: `id`, `chrom`, `pos`, `ref`, and `alt`.
        - The genotype information is stored in `mdata.uns[genotype_key]` as a pandas DataFrame where rows correspond to SNPs and columns correspond to samples.

    Returns
    -------
        The function modifies `mdata` in place by adding variant and genotype information.

    Raises
    ------
    AssertionError
        If a SNP with multiple alternative alleles is encountered.

    Examples
    --------
    >>> import muon as mu
    >>> import cell2net as cn
    >>> mdata = mu.MuData({})
    >>> cn.pp.add_genomic_variants(mdata, "variants.vcf")
    >>> mdata.uns["variants"].head()
    >>> mdata.uns["genotype"].head()
    """
    reader = vcfpy.Reader.from_path(vcf_file)

    sample_ids = reader.header.samples.names  # type: ignore
    sample_ids = [str(x) for x in sample_ids]  # Ensure all elements are strings

    snp_ids, snp_chroms, snp_positions, snp_refs, snp_alts = [], [], [], [], []
    genotypes = []
    for record in reader:
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

    # Add SNP information to mdata
    mdata.uns[variants_key] = pd.DataFrame(
        data={
            "id": snp_ids,
            "chrom": snp_chroms,
            "pos": snp_positions,
            "ref": snp_refs,
            "alt": snp_alts,
        }
    )

    # Add donor information to mdata
    mdata.uns[genotype_key] = pd.DataFrame(
        data=genotypes, columns=sample_ids, index=snp_ids
    ).astype("Int8")

    return None


def variant_to_peak(
    mdata: MuData,
    mod_name: str = "atac",
    variants_key: str = "variants",
    peak_key: str = "peak",
) -> None:

    mdata["atac"].var["peak"] = (
        mdata["atac"].var["chrom"]
        + ":"
        + mdata["atac"].var["start"].astype(str)
        + "-"
        + mdata["atac"].var["end"].astype(str)
    )

    pass
