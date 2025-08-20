from pathlib import Path

import numpy as np
import pandas as pd
import vcfpy
from mudata import MuData
from tqdm.auto import tqdm

from cell2net._logging import logger


def _get_genomic_variants(vcf_file: str | Path, chrom: str, start: int, end: int):
    """
    Extracts SNP (single nucleotide polymorphism) information and genotypes from a VCF file within a specified genomic region.

    This function utilizes the `vcfpy` library to read and filter VCF records based on the provided genomic coordinates.

    Parameters
    ----------
    vcf_file :
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
        - `donor`: Donor ID found in the VCF file.
        - `genotype`: Genotype of the donor for the SNP, encoded as:
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
    reader = vcfpy.Reader.from_path(vcf_file)

    sample_ids = reader.header.samples.names  # type: ignore
    sample_ids = [str(x) for x in sample_ids]  # Ensure all elements are strings

    records = reader.fetch(chrom, start, end)

    snp_ids, snp_chroms, snp_positions, snp_refs, snp_alts = [], [], [], [], []
    genotypes = []
    for record in records:
        if record is None or not record.is_snv():
            continue

        assert len(record.ALT) == 1, f"find multiple alternatives for a SNP {record.ID[0]} "

        snp_ids.append(record.ID[0])
        snp_chroms.append(record.CHROM)
        snp_positions.append(record.POS)
        snp_refs.append(record.REF)
        snp_alts.append(record.ALT[0].value)

        # Extract genotype information
        genotype = [call.data.get("GT") or "./." for call in record.calls]
        genotype = [str(x) for x in genotype]  # Ensure all elements are strings
        genotype = [0 if x == "0/0" else 1 if x == "0/1" else 2 if x == "1/1" else np.nan for x in genotype]

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
    df_genotype = pd.DataFrame(data=genotypes, columns=sample_ids, index=snp_ids).astype("Int8")

    df = pd.merge(df_snp, df_genotype, left_index=True, right_index=True)
    df["snp_id"] = df.index

    df = pd.melt(
        df,
        id_vars=["snp_id", "chrom", "pos", "ref", "alt"],
        var_name="donor",
        value_name="genotype",
    )
    df["snp_id"] = df["snp_id"].astype(str)
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(int)
    df["ref"] = df["ref"].astype(str)
    df["alt"] = df["alt"].astype(str)

    df = df[df['ref'].isin(['A', 'C', 'G', 'T']) & df['alt'].isin(['A', 'C', 'G', 'T'])]

    return df


def get_genomic_variants(
    mdata: MuData,
    donor_col_key: str,
    vcf_file: str | Path,
    n_cpus: int = 1,
    atac_mod: str = "atac"
) -> pd.DataFrame:
    """
    Extract and annotate genomic variants from a VCF file that overlap with ATAC-seq peaks.

    This function identifies single nucleotide variants (SNVs) from a VCF file that fall within
    the genomic regions defined by ATAC-seq peaks. It processes each peak region, extracts
    overlapping variants, and returns a comprehensive DataFrame containing variant information
    along with genotype data for each donor/sample.

    Parameters
    ----------
    mdata : MuData
        A MuData object containing the ATAC-seq modality with peak information stored in
        `mdata[atac_mod].uns['peaks']`. The peaks DataFrame should contain columns:
        'chr', 'start', 'end' defining genomic coordinates.
    donor_col_key : str
        Column name in `mdata[atac_mod].obs` that specifies donor/sample identities.
        Used to filter variants to only those present in the experimental samples.
        Variants for donors not present in this column will be excluded.
    vcf_file : str or Path
        Path to a coordinate-sorted VCF file containing genomic variant data.
        The file should be compatible with the `vcfpy` library for reading.
    n_cpus : int, default 1
        Number of CPU cores to use for parallel processing. If > 1, peak processing
        will be parallelized using multiprocessing to speed up variant extraction.
    atac_mod : str, default "atac"
        Name of the modality in `mdata` that contains the ATAC-seq data and peak
        information. Must exist in `mdata.mod_names`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing variant information with the following columns:

        - `snp_id` : str - Variant identifier from the VCF file
        - `chrom` : str - Chromosome name (e.g., 'chr1', '1')
        - `pos` : int - Genomic position (1-based coordinate)
        - `ref` : str - Reference allele (A, C, G, or T)
        - `alt` : str - Alternative allele (A, C, G, or T)
        - `peak` : str - Peak identifier where the variant was found
        - `donor` : str - Donor/sample identifier
        - `genotype` : int - Encoded genotype:
            * 0 = homozygous reference (0/0)
            * 1 = heterozygous (0/1)
            * 2 = homozygous alternative (1/1)
            * NaN = missing genotype

    Raises
    ------
    ValueError
        If the specified `atac_mod` is not found in `mdata.mod_names`.
    KeyError
        If `donor_col_key` is not found in `mdata[atac_mod].obs.columns`.
    AssertionError
        If any variant in the VCF file has multiple alternative alleles.

    Notes
    -----
    - Only single nucleotide variants (SNVs) with exactly one alternative allele are processed.
      Indels and multiallelic variants are automatically skipped.
    - Variants are filtered to include only those with reference and alternative alleles
      in the set {A, C, G, T}.
    - Duplicate variants (same SNP ID and donor) are automatically removed, keeping the first occurrence.
    - When the same donor has multiple genotypes for the same genomic position, only the first
      genotype is retained after grouping by chromosome, position, reference allele, donor, and peak.
    - The VCF file must be coordinate-sorted for efficient processing.
    - Peak regions are defined by the 'chr', 'start', and 'end' columns in the peaks DataFrame.

    Examples
    --------
    Extract variants overlapping with ATAC-seq peaks:

    >>> import mudata as md
    >>> import cell2net as cn
    >>>
    >>> # Load multimodal data with ATAC-seq peaks
    >>> mdata = md.read_h5mu("multiome_data.h5mu")
    >>>
    >>> # Extract variants using single CPU
    >>> variants_df = cn.pp.get_genomic_variants(
    ...     mdata=mdata,
    ...     donor_col_key="donor_id",
    ...     vcf_file="variants.vcf.gz"
    ... )
    >>>
    >>> # Use parallel processing for faster execution
    >>> variants_df = cn.pp.get_genomic_variants(
    ...     mdata=mdata,
    ...     donor_col_key="sample_id",
    ...     vcf_file="large_variants.vcf.gz",
    ...     n_cpus=4
    ... )
    >>>
    >>> print(f"Found {len(variants_df)} variants across {variants_df['peak'].nunique()} peaks")
    """
    logger.info("Processing variants started!")
    if atac_mod not in mdata.mod_names:
        logger.error(f"Cannot find modality: {atac_mod}")
        return None

    adata = mdata[atac_mod]

    if donor_col_key not in adata.obs.columns:
        logger.error(f"Cannot find column: {donor_col_key}")
        return None

    df_peaks = adata.uns['peaks']
    df_var_list = []

    logger.info(f"Found {len(df_peaks)} peaks in {atac_mod} modality.")

    if n_cpus > 1:
        from multiprocessing import Pool

        # prepare agument for parallel processing
        args = []
        for chrom, start, end in zip(
            df_peaks['chr'],
            df_peaks['start'],
            df_peaks['end'],
            strict=False,
        ):
            args.append((vcf_file, chrom, start, end))

        # use multiprocessing to speed up the process
        logger.info(f"Using {n_cpus} CPUs for parallel processing.")
        with Pool(n_cpus) as pool:
            results = pool.starmap(_get_genomic_variants, tqdm(args, desc="Processing variants"))

        # combine results
        for i, df_var in enumerate(results):
            df_var["peak"] = df_peaks.index[i]
            df_var_list.append(df_var)
        logger.info("Finished processing variants in parallel.")
    else:
        # process in serial
        logger.info("Processing variants in serial.")
        for i, (chrom, start, end) in enumerate(
            tqdm(
                zip(
                    df_peaks['chr'],
                    df_peaks['start'],
                    df_peaks['end'],
                    strict=False,
                ),
                total=len(df_peaks),
                desc="Processing variants",
            )
        ):
            df_var = _get_genomic_variants(vcf_file, chrom, start, end)
            df_var["peak"] = df_peaks.index[i]
            df_var_list.append(df_var)

    df_var = pd.concat(df_var_list, ignore_index=True)

    logger.info(f"Found {len(df_var)} variants in total across all peaks.")

    # filter out samples not in the adata object
    logger.info(f"Filtering variants for donors in {donor_col_key} column.")
    donors = adata.obs[donor_col_key].unique()
    df_var = df_var[df_var[donor_col_key].isin(donors)]

    logger.info(f"Obtained {len(df_var)} variants after filtering.")

    df_var = df_var[[donor_col_key, 'peak', 'snp_id', 'chrom', 'pos', 'ref', 'alt', 'genotype']]
    df_var = df_var.sort_values([donor_col_key, 'peak', 'snp_id'])

    # # remove duplicates using groupby (faster than drop_duplicates)
    # logger.info("Removing duplicate variants.")
    # df_var = df_var.groupby(["snp_id", donor_col_key], as_index=False).first()

    # # sometimes the same donor has multiple genotypes for the same SNP
    # # in this case, we take the first one
    # df_var = df_var.groupby(["chrom", "pos", "ref", donor_col_key, "peak"]).first().reset_index()

    logger.info(f"Found {len(df_var)} variants in total.")
    logger.info("Processing variants finished!")

    return df_var
