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
    logger.info(f"Filtering variants for donors.")
    donors = adata.obs[donor_col_key].unique()
    df_var = df_var[df_var[donor_col_key].isin(donors)]

    logger.info(f"Obtained {len(df_var)} variants after filtering.")

    # remove duplicates using groupby (faster than drop_duplicates)
    logger.info("Removing duplicate variants.")
    df_var = df_var.sort_values(by=[donor_col_key, "snp_id"])
    df_var = df_var.groupby([donor_col_key, "snp_id"], as_index=False).first()

    # # sometimes the same donor has multiple genotypes for the same SNP
    # # in this case, we take the first one
    df_var = df_var.sort_values(by=[donor_col_key, "chrom", "pos", "ref", "peak"])
    df_var = df_var.groupby([donor_col_key, "chrom", "pos", "ref", "peak"], as_index=False).first()

    logger.info("Sorting variants.")
    df_var = df_var[[donor_col_key, 'peak', 'snp_id', 'chrom', 'pos', 'ref', 'alt', 'genotype']]
    df_var = df_var.sort_values([donor_col_key, 'peak', 'snp_id']).reset_index(drop=True)

    logger.info(f"Found {len(df_var)} variants in total.")
    logger.info("Processing variants finished!")

    return df_var


def _add_variants_to_sequence(
    df_seq: pd.DataFrame,
    df_var: pd.DataFrame
) -> pd.DataFrame:
    """
    Update reference DNA sequences with genomic variants based on genotype information.

    This function modifies a pair of haplotype sequences (`seq_1` and `seq_2`) stored in `df_seq`,
    using variant data from `df_variants`. For each variant, the reference allele is checked
    against the sequence at the given position, and if matched, the sequence is updated
    depending on the genotype:

    - Genotype 1 (heterozygous): update `seq_2` with the ALT allele
    - Genotype 2 (homozygous alt): update both `seq_1` and `seq_2` with the ALT allele

    Parameters
    ----------
    df_seq :
        A DataFrame containing the reference sequences for each (peak, donor) pair.
            Must include the following columns:
            - 'peak': unique peak identifier
            - 'donor': donor identifier
            - 'seq_1': reference haplotype 1 sequence
            - 'seq_2': reference haplotype 2 sequence
            - 'start': start genomic coordinate of the peak
    df_var :
        A DataFrame containing variant information to apply.
            Must include the following columns:
            - 'donor': donor ID
            - 'peak': peak ID the variant overlaps
            - 'snp_id': SNP id
            - 'chrom': chromosome
            - 'pos': genomic position of the variant (1-based)
            - 'ref': reference allele
            - 'alt': alternate allele
            - 'genotype': genotype code (1 for het, 2 for hom-alt)

    Returns
    -------
        A new DataFrame with the same structure as `df_seq`,
        but with updated `seq_1` and `seq_2` sequences based on the input variants.

    Raises
    ------
    AssertionError
        If the reference base in the sequence does not match the provided 'ref' allele at the variant position.

    Notes
    -----
        - Assumes `df_seq` is uniquely indexed by ('peak', 'sample').
        - Assumes all positions in `df_variants` fall within the corresponding peak interval.
        - Index is temporarily set during processing and restored at the end.
    """
    df_seq = df_seq.set_index(["peak", "donor"])

    # Update the sequences with variants
    for _, row in tqdm(df_var.iterrows(), total=len(df_var)):
        peak, donor, genotype = row["peak"], row["donor"], row["genotype"]

        start = df_seq.loc[(peak, donor)]["start"]
        seq_1 = df_seq.loc[(peak, donor)]["seq_1"]
        seq_2 = df_seq.loc[(peak, donor)]["seq_2"]

        # check if the position is within the sequence
        assert (
            seq_1[row["pos"] - start - 1] == row["ref"]
        ), f"peak: {peak}, donor: {donor}, ref: {row['ref']}, seq_1: {seq_1[row['pos'] - start - 1]}, pos: {row['pos']}, start: {start}"
        assert (
            seq_2[row["pos"] - start - 1] == row["ref"]
        ), f"peak: {peak}, donor: {donor}, ref: {row['ref']}, seq_2: {seq_2[row['pos'] - start - 1]}, pos: {row['pos']}, start: {start}"

        if genotype == 1:
            df_seq.loc[(peak, donor), "seq_2"] = (
                seq_2[: row["pos"] - start - 1]
                + row["alt"]
                + seq_2[row["pos"] - start :]
            )

        elif genotype == 2:
            df_seq.loc[(peak, donor), "seq_1"] = (
                seq_1[: row["pos"] - start - 1]
                + row["alt"]
                + seq_1[row["pos"] - start :]
            )
            df_seq.loc[(peak, donor), "seq_2"] = (
                seq_2[: row["pos"] - start - 1]
                + row["alt"]
                + seq_2[row["pos"] - start :]
            )
        else:
            logger.error(
                f"Unknown genotype: {genotype}, peak: {peak}, donor: {donor}"
            )
            continue

    df_seq = df_seq.reset_index()

    return df_seq


def add_variants_to_sequence(
    mdata: MuData,
    df_var: pd.DataFrame,
    donor_col_key: str,
    atac_mod: str = "atac",
    personal_genome_seq: str = "personal_genome_seq",
    n_cpus: int = 1,
    verbose: bool = False,
) -> None:
    """
    Add genomic variants to DNA sequences from peak regions to generate personalized haplotype sequences.

    This function takes a MuData object containing ATAC-seq data with reference DNA sequences for peak regions,
    and variant information, then applies the variants to the sequences per donor to produce haplotype-specific
    (seq_1 and seq_2) updated sequences reflecting individual genotypes. The function supports both single-core
    and multi-core processing for improved performance on large datasets.

    Parameters
    ----------
    mdata : MuData
        A MuData object containing the ATAC-seq modality with peak information stored in
        `mdata[atac_mod].uns['peaks']`. The peaks DataFrame should contain columns:
        'chr', 'start', 'end', 'sequence' defining genomic coordinates and reference sequences.
    df_var : pd.DataFrame
        A DataFrame containing variant information with the following required columns:

        - `donor` : str - Donor/sample identifier (must match `donor_col_key`)
        - `peak` : str - Peak identifier where the variant was found
        - `snp_id` : str - Variant identifier from the VCF file
        - `chrom` : str - Chromosome name
        - `pos` : int - Genomic position (1-based coordinate)
        - `ref` : str - Reference allele
        - `alt` : str - Alternative allele
        - `genotype` : int - Encoded genotype (1=heterozygous, 2=homozygous alternative)
    donor_col_key : str
        Column name in `mdata[atac_mod].obs` that specifies donor/sample identities.
        Must match the 'donor' column in `df_var`.
    atac_mod : str, default "atac"
        Name of the modality in `mdata` that contains the ATAC-seq data and peak information.
        Must exist in `mdata.mod_names`.
    seq_with_variants_key : str, default "seq_with_variants"
        The key under which to store the resulting DataFrame in `mdata[atac_mod].uns`,
        containing haplotype-aware sequences with applied variants.
    n_cpus : int, default 1
        Number of CPU cores to use for parallel processing. If > 1, uses multiprocessing
        to parallelize processing across donors for improved performance.

    Returns
    -------
    None
        The function modifies `mdata` in place by storing the resulting DataFrame with
        variant-updated sequences in `mdata[atac_mod].uns[seq_with_variants_key]`.

        The stored DataFrame contains:

        - `peak` : str - Peak identifier
        - `donor` : str - Donor/sample identifier
        - `seq_1` : str - Haplotype 1 sequence with variants applied
        - `seq_2` : str - Haplotype 2 sequence with variants applied

    Raises
    ------
    ValueError
        If the specified `atac_mod` is not found in `mdata.mod_names`.
    AssertionError
        If the reference allele in the sequence does not match the variant's reference
        base at the specified position, indicating inconsistent data.

    Notes
    -----
    - **Haplotype handling**: Each peak-donor combination gets two sequences (seq_1, seq_2)
      representing the two chromosome copies (haplotypes).
    - **Variant application**:

      * Genotype 0 (homozygous reference): no changes applied
      * Genotype 1 (heterozygous): only seq_2 is updated with the alternative allele
      * Genotype 2 (homozygous alternative): both seq_1 and seq_2 are updated

    - **Coordinate systems**: DNA sequences are 0-based Python strings while variant
      positions are 1-based genomic coordinates.
    - **Performance**: Multi-core processing significantly improves performance for
      large datasets by parallelizing across donors.
    - **Memory usage**: The function creates a full grid of all peak-donor combinations,
      which may require substantial memory for large datasets.
    - **Validation**: The function validates that reference alleles in variants match
      the corresponding positions in the reference sequences.

    Examples
    --------
    Apply variants to sequences using single CPU:

    >>> import mudata as md
    >>> import cell2net as cn
    >>>
    >>> # Load data and get variants
    >>> mdata = md.read_h5mu("multiome_data.h5mu")
    >>> variants_df = cn.pp.get_genomic_variants(
    ...     mdata=mdata,
    ...     donor_col_key="donor_id",
    ...     vcf_file="variants.vcf.gz"
    ... )
    >>>
    >>> # Apply variants to sequences
    >>> cn.pp.add_variants_to_sequence(
    ...     mdata=mdata,
    ...     df_var=variants_df,
    ...     donor_col_key="donor_id"
    ... )
    >>>
    >>> # Access variant-updated sequences
    >>> seq_df = mdata["atac"].uns["seq_with_variants"]
    >>> print(f"Generated {len(seq_df)} haplotype sequences")

    Use parallel processing for better performance:

    >>> cn.pp.add_variants_to_sequence(
    ...     mdata=mdata,
    ...     df_var=variants_df,
    ...     donor_col_key="sample_id",
    ...     n_cpus=4,
    ...     seq_with_variants_key="personalized_sequences"
    ... )

    Check the results:

    >>> seq_df = mdata["atac"].uns["seq_with_variants"]
    >>> print(f"Donors: {seq_df['donor'].nunique()}")
    >>> print(f"Peaks: {seq_df['peak'].nunique()}")
    >>> print(f"Total sequences: {len(seq_df)}")
    """
    logger.info("Adding variants started!")
    if atac_mod not in mdata.mod_names:
        logger.error(f"Cannot find modality: {atac_mod}")
        return None

    adata = mdata[atac_mod]

    df_peaks = adata.uns['peaks']
    df_peaks["peak"] = df_peaks.index
    df_peaks = df_peaks.reset_index(drop=True)

    donor_list = adata.obs[donor_col_key].unique()

    # create dataframe for peaks and donors
    # assume that seq_1 is for chromatid 1 and seq_2 is for chromatid 2
    logger.info(
        f"Creating dataframe for all {len(df_peaks)} peaks and {len(donor_list)} donors"
    )
    df_seq = pd.DataFrame(
        columns=["peak", "donor", "seq_1", "seq_2"],
        index=range(len(df_peaks) * len(donor_list)),
    )
    df_seq["peak"] = np.repeat(list(df_peaks["peak"]), len(donor_list))
    df_seq["donor"] = np.tile(donor_list, len(df_peaks))
    df_seq["start"] = np.repeat(list(df_peaks["start"]), len(donor_list))
    df_seq["seq_1"] = np.repeat(df_peaks['sequence'].tolist(), len(donor_list))
    df_seq["seq_2"] = np.repeat(df_peaks['sequence'].tolist(), len(donor_list))

    # update the sequences with variants
    # only update the sequences with heterozygous and homozygous alternate genotypes
    # logger.info("Keep genotypes with alternative alleles")
    df_var = df_var[df_var["genotype"].isin([1, 2])].reset_index(drop=True)

    # logger.info(f"Number of variants with donors: {len(df_var)}")
    if n_cpus == 1:
        df_seq_list = []
        for donor in tqdm(donor_list, desc="Updating sequences with variants"):
            _df_seq = df_seq[df_seq["donor"] == donor].reset_index(drop=True).copy()
            _df_var = df_var[df_var["donor"] == donor].reset_index(drop=True).copy()

            # Update sequence
            _df_seq = _add_variants_to_sequence(_df_seq, _df_var)
            df_seq_list.append(_df_seq)

        df_seq = pd.concat(df_seq_list, ignore_index=True)

    else:
        logger.info(f"Using {n_cpus} CPUs for parallel processing.")
        from multiprocessing import Pool

        # split the df_var by donor
        # and run the update_sequence_with_variants in parallel
        args = []
        for donor in donor_list:
            _df_seq = df_seq[df_seq["donor"] == donor].reset_index(drop=True).copy()
            _df_var = df_var[df_var["donor"] == donor].reset_index(drop=True).copy()

            args.append((_df_seq, _df_var))

        # run the _add_variants_to_sequence in parallel
        with Pool(n_cpus) as pool:
            results = pool.starmap(
                _add_variants_to_sequence,
                tqdm(args, desc="Updating sequences with variants"),
            )

        # combine the results
        df_seq = pd.concat(results, ignore_index=True)

    df_seq = df_seq.drop(columns=["start"])
    df_seq = df_seq.sort_values(by=["peak", "donor"]).reset_index(drop=True)

    adata.uns[personal_genome_seq] = df_seq
    logger.info("Adding variants finished!")

    return None
