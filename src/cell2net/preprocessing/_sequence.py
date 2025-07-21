"""Functions to process DNA sequences"""

import numpy as np
import pandas as pd
from mudata import MuData
from pysam import FastaFile
from tqdm.auto import tqdm
import torch
from cell2net._logging import logger

def random_seq(seq_len: int, bases: list[str] | None = None) -> str:
    """
    Generate a random nucleotide sequence of a specified length.

    This function creates a random sequence of nucleotides (or other bases)
    by selecting characters from a provided list of bases. If no base list
    is provided, the default bases are adenine (A), cytosine (C), guanine (G),
    and thymine (T).

    Parameters
    ----------
    seq_len : int
        The length of the random sequence to generate.
    bases : list[str], optional
        A list of characters to use as the base set for the sequence.
        Defaults to ["A", "C", "G", "T"].

    Returns
    -------
    str
        A randomly generated sequence of the specified length using the
        provided bases.

    Examples
    --------
    >>> random_seq(10)
    'ACGTGCTAGC'
    >>> random_seq(5, bases=["A", "T"])
    'TATTA'
    """
    if bases is None:
        bases = ["A", "C", "G", "T"]

    rand_seq = "".join([np.random.choice(bases) for i in range(seq_len)])
    return rand_seq


def seq_to_one_hot(seq: str) -> np.ndarray:
    """
    One-hot encodes a DNA sequence while handling unknown bases.

    Only ACTGN allow.

    Parameters
    ----------
    seq : str
            A DNA sequence.

    Returns
    -------
        One-hot encoded matrix with shape (sequence_length, 4).
    """
    seq = seq.upper()

    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        logger.error(
            f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}"
        )

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],  # for unknown bases
    }

    # Create array from nucleotide sequence
    one_hot = np.array([nuc_d[x] for x in seq], dtype=np.float32)

    return one_hot

def one_hot_to_seq(one_hot: np.ndarray) -> str:
    """
    Converts a one-hot encoded DNA matrix back to a nucleotide sequence.

    Parameters
    ----------
    one_hot :
        A NumPy array of shape (sequence_length, 4), where each row represents
        a nucleotide in one-hot encoding format.

    Returns
    -------
        The reconstructed DNA sequence, where each character represents a nucleotide.

    Notes
    -----
        - If a row has all zeros, the function assigns "N" to represent an unknown nucleotide.
        - The function assumes a valid one-hot encoding where each row has at most one "1".

    Examples
    --------
    Convert a one-hot encoded DNA sequence back to a string:

    >>> import numpy as np
    >>> one_hot = np.array([
    ...     [1, 0, 0, 0],  # A
    ...     [0, 1, 0, 0],  # C
    ...     [0, 0, 0, 0],  # N (unknown base)
    ...     [0, 0, 1, 0],  # G
    ...     [0, 0, 0, 1]   # T
    ... ])
    >>> one_hot_to_seq(one_hot)
    'ACNGT'
    """
    # Dictionary returning nucleotide for each one-hot encoding
    mapping = {0: "A", 1: "C", 2: "G", 3: "T"}

    # Convert one-hot encoding to indices
    nucleotide_indices = np.argmax(one_hot, axis=1)

    # Handle unknown bases (if all zeros in one-hot row)
    sequence = "".join(
        mapping[i] if one_hot[row].sum() == 1 else "N"
        for row, i in enumerate(nucleotide_indices)
    )

    return sequence


def dinucleotide_shuffle_str(seq: str, random_state: int = 42) -> str:
    """
    Shuffle a DNA sequence while preserving its dinucleotide composition.

    This function takes a DNA sequence as input, splits it into overlapping
    dinucleotides, shuffles them, and reconstructs a sequence with the same
    dinucleotide composition but in a randomized order.

    Parameters
    ----------
    seq :
        The DNA sequence to shuffle. Must be a string of nucleotides (e.g., "ATCG").
        Sequences with fewer than 2 characters are returned unchanged.

    random_state :
        The randome state.

    Returns
    -------
        A shuffled version of the input sequence with the same dinucleotide composition.
        If the input sequence has fewer than 2 characters, it is returned as is.

    Notes
    -----
        - The function ensures that the dinucleotide composition of the shuffled sequence matches that of the input sequence, but the overall sequence order is randomized.
        - Randomization is achieved using the `random.shuffle` function.

    Examples
    --------
    >>> import cell2net as cn
    >>> cn.pp.dinucleotide_shuffle_str("ATCG")
    'TACG'
    >>> cn.pp.dinucleotide_shuffle_str("A")
    'A'
    >>> cn.pp.dinucleotide_shuffle_str("")
    ''
    """
    if len(seq) < 2:
        return seq

    # Create a list of dinucleotides
    dinucleotides = [seq[i : i + 2] for i in range(len(seq) - 1)]

    # Shuffle the dinucleotides
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(dinucleotides)

    # Reconstruct the sequence from shuffled dinucleotides
    shuffled_sequence = dinucleotides[0]
    for dinucleotide in dinucleotides[1:]:
        shuffled_sequence += dinucleotide[1]

    return shuffled_sequence


# def dinucleotide_shuffle(
#     dna_seq: np.ndarray | str, return_str: bool = True
# ) -> np.ndarray | str:
#     """
#     Shuffle DNA sequence while preserving its dinucleotide composition

#     Parameters
#     ----------
#     seq : np.ndarray | str
#         _description_

#     Returns
#     -------
#     np.ndarray | str
#         _description_
#     """
#     if isinstance(dna_seq, np.ndarray):
#         dna_seq = one_hot_to_seq(dna_seq)

#     if len(dna_seq) < 2:
#         return dna_seq

#     if isinstance(dna_seq, np.ndarray):
#         seq = one_hot_to_seq(seq)

#         if len(seq) < 2:
#             return seq

#     elif isinstance(dna_seq, str):
#         shuffled_sequence = dinucleotide_shuffle_str(dna_seq)


def dinucleotide_shuffle_one_hot(one_hot: np.ndarray) -> np.ndarray:
    """
    Shuffle a one-hot encoded DNA sequence while preserving its dinucleotide composition.

    This function converts a one-hot encoded DNA sequence into its nucleotide representation,
    shuffles it while maintaining the same dinucleotide composition, and then converts the
    shuffled sequence back into one-hot encoding.

    Parameters
    ----------
    one_hot:
        A 2D array of shape (L, 4), where L is the sequence length, and each row is a
        one-hot encoded nucleotide. Each row should contain exactly one 1 and three 0s,
        corresponding to the nucleotides "A", "C", "G", and "T".

    Returns
    -------
        A 2D array of shape (L, 4) representing the shuffled sequence in one-hot encoding.
        The dinucleotide composition of the original sequence is preserved.

    Notes
    -----
        - The function assumes the input sequence is valid one-hot encoding. Behavior is undefined if the input contains invalid rows.
        - Shuffling is performed on the nucleotide sequence derived from the one-hot input, and the shuffled sequence is converted back to one-hot encoding.
        - The function uses the dinucleotide_shuffle helper function to handle the shuffling of the nucleotide sequence.

    Examples
    --------
    >>> import numpy as np
    >>> import cell2net as cn
    >>> import random
    >>> random.seed(42)
    >>> one_hot_sequence = np.array([
    ...     [1, 0, 0, 0],  # A
    ...     [0, 1, 0, 0],  # C
    ...     [0, 0, 1, 0],  # G
    ...     [0, 0, 0, 1]   # T
    ... ])
    >>> shuffled_one_hot = cn.pp.dinucleotide_one_hot_shuffle(one_hot_sequence)
    >>> shuffled_one_hot
    array([[0., 1., 0., 0.],  # "C"
           [1., 0., 0., 0.],  # "A"
           [0., 0., 0., 1.],  # "T"
           [0., 0., 1., 0.]]) # "G"
    """
    # Convert one-hot encoded sequence to nucleotide sequence
    seq = one_hot_to_seq(one_hot)
    shuffled_sequence = dinucleotide_shuffle_str(seq)
    shuffled_one_hot = seq_to_one_hot(shuffled_sequence)

    return shuffled_one_hot


def add_dna_sequence(
    mdata: MuData,
    ref_fasta: str,
    mod_name: str = "atac",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    sequence_var_key: str = "dna_sequence",
) -> None:
    """
    Add sequences to peak metadata in a MuData object.

    This function retrieves DNA sequences for genomic regions specified in the `.var`
    attribute of the AnnData object within a MuData object. The sequences are fetched
    from a reference FASTA file and added as metadata under the specified key.

    Parameters
    ----------
    mdata :
        A MuData object containing the modality with peak metadata.
    ref_fasta :
        Path to the reference FASTA file. This file must be indexed (e.g., with samtools faidx).
    mod_name :
        The name of the modality containing peak data. Defaults to "atac".
    chr_var_key :
        The key in `.var` that contains chromosome names. Defaults to "chr".
    start_var_key :
        The key in `.var` that contains the start positions of peaks. Defaults to "start".
    end_var_key :
        The key in `.var` that contains the end positions of peaks. Defaults to "end".
    sequence_var_key :
        The key under which the retrieved DNA sequences will be stored in `.var`. Defaults to "dna_sequence".

    Returns
    -------
    None
        The function modifies the MuData object in place by adding DNA sequences to the
        specified key in the `.var` attribute.

    Raises
    ------
    AssertionError
        If the specified modality (`mod_name`) is not found in the MuData object.
    FileNotFoundError
        If the `ref_fasta` file does not exist or is not properly indexed.

    Examples
    --------
    >>> from mudata import MuData
    >>> import anndata as ad
    >>> import pandas as pd
    >>> import cell2net as cn
    >>> data = ad.AnnData(var=pd.DataFrame({
    ...     "chr": ["chr1", "chr2"],
    ...     "start": [100, 200],
    ...     "end": [150, 250]
    ... }))
    >>> mdata = MuData({"atac": data})
    >>> cn.pp.add_dna_sequence(mdata, ref_fasta="reference.fasta")
    >>> print(mdata["atac"].var["dna_sequence"])
    0    ATCGTTGAC...
    1    TGGCCAATA...
    """
    assert mod_name in mdata.mod_names, f"Cannot find modality: {mod_name}"
    adata = mdata[mod_name]

    fasta = FastaFile(filename=ref_fasta)
    df = adata.var[[chr_var_key, start_var_key, end_var_key]]

    seqs = []
    for chrom, start, end in tqdm(
        zip(
            df[chr_var_key],
            df[start_var_key],
            df[end_var_key],
            strict=False,
        ),
        desc="Fetching sequences",
        total=len(df),
    ):
        seqs.append(fasta.fetch(chrom, start, end).upper())

    adata.var[sequence_var_key] = seqs

    return None


def update_sequence_with_variants(
    df_seq: pd.DataFrame, df_variants: pd.DataFrame
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
        A DataFrame containing the reference sequences for each (peak, sample) pair.
            Must include the following columns:
            - 'peak': unique peak identifier
            - 'sample': sample identifier
            - 'start': start genomic coordinate of the peak
            - 'seq_1': reference haplotype 1 sequence
            - 'seq_2': reference haplotype 2 sequence
    df_variants :
        A DataFrame containing variant information to apply.
            Must include the following columns:
            - 'peak': peak ID the variant overlaps
            - 'sample': sample ID
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
    df_seq = df_seq.set_index(["peak", "sample"])

    # Update the sequences with variants
    for _, row in tqdm(df_variants.iterrows(), total=len(df_variants)):
        peak, sample, genotype = row["peak"], row["sample"], row["genotype"]

        start = df_seq.loc[(peak, sample)]["start"]
        seq_1 = df_seq.loc[(peak, sample)]["seq_1"]
        seq_2 = df_seq.loc[(peak, sample)]["seq_2"]

        # check if the position is within the sequence
        assert (
            seq_1[row["pos"] - start - 1] == row["ref"]
        ), f"peak: {peak}, sample: {sample}, ref: {row['ref']}, seq_1: {seq_1[row['pos'] - start - 1]}, pos: {row['pos']}, start: {start}"
        assert (
            seq_2[row["pos"] - start - 1] == row["ref"]
        ), f"peak: {peak}, sample: {sample}, ref: {row['ref']}, seq_2: {seq_2[row['pos'] - start - 1]}, pos: {row['pos']}, start: {start}"

        if genotype == 1:
            df_seq.loc[(peak, sample), "seq_2"] = (
                seq_2[: row["pos"] - start - 1]
                + row["alt"]
                + seq_2[row["pos"] - start :]
            )

        elif genotype == 2:
            df_seq.loc[(peak, sample), "seq_1"] = (
                seq_1[: row["pos"] - start - 1]
                + row["alt"]
                + seq_1[row["pos"] - start :]
            )
            df_seq.loc[(peak, sample), "seq_2"] = (
                seq_2[: row["pos"] - start - 1]
                + row["alt"]
                + seq_2[row["pos"] - start :]
            )
        else:
            logger.error(
                f"Unknown genotype: {genotype}, peak: {peak}, sample: {sample}"
            )
            continue

    df_seq = df_seq.reset_index()

    return df_seq


def add_variants_to_sequence(
    mdata: MuData,
    atac_mod: str = "atac",
    n_cpus: int = 1,
    sample_col_key: str = "bestSample",
    sequence_var_key: str = "dna_sequence",
    variants_key: str = "variants",
    seq_with_variants_key: str = "seq_with_variants",
    inplace: bool = True,
) -> None | pd.DataFrame:
    """
    Add genomic variants to DNA sequences from peak regions to generate personalized haplotype sequences.

    This function takes a MuData object containing ATAC-seq data, reference DNA sequences for peak regions,
    and variant information. It applies the variants to the sequences per sample to produce
    haplotype-specific (seq_1 and seq_2) updated sequences reflecting individual genotypes.
    Supports single-core or multi-core processing.

    Parameters
    ----------
    mdata :
        A MuData object containing the modality with ATAC-seq data and variant information.
    atac_mod :
         Name of the modality in `mdata` that contains the ATAC-seq data.
    n_cpus :
        Number of CPU cores to use. If >1, uses multiprocessing to parallelize across samples.
    sample_col_key :
        The name of the column in `adata.obs` that identifies sample IDs.
    sequence_var_key :
        The name of the column in `adata.var` that contains the reference DNA sequence for each peak.
    variants_key :
        The key in `adata.uns` that stores the variant information (as a DataFrame), including columns:
        - 'peak': peak ID
        - 'sample': sample ID
        - 'pos': variant position
        - 'ref': reference allele
        - 'alt': alternate allele
        - 'genotype': genotype (1 for het, 2 for hom-alt)
    seq_with_variants_key :
        The key under which to store the resulting DataFrame in `adata.uns`, containing haplotype-aware sequences.
    inplace :
        If True, the resulting DataFrame is stored in `adata.uns[seq_with_variants_key]`.
        If False, the function returns the DataFrame directly.

    Returns
    -------
        Returns `None` if `inplace=True`.
        Otherwise, returns a DataFrame with updated haplotype sequences:
        - 'peak': peak ID
        - 'sample': sample ID
        - 'seq_1': sequence with genotype 1 or 2 applied to haplotype 1
        - 'seq_2': sequence with genotype 1 or 2 applied to haplotype 2

    Notes
    -----
        - Each variant is applied to its corresponding peak and sample-specific sequence.
        - Assumes DNA sequences are 0-based Python strings and variant positions are 1-based.
        - For heterozygous (1) genotypes, only `seq_2` is updated.
        - For homozygous alternate (2) genotypes, both `seq_1` and `seq_2` are updated.
        - Peaks and sample combinations are expanded into a full grid for processing.
        - Performance can be improved with parallel execution using multiple CPUs.
        - Requires the helper function `update_sequence_with_variants()`.

    Raises
    ------
    Logs errors if:
        - The specified modality or keys are not found in the `MuData` object.
        - The reference allele in the sequence does not match the variant's reference base.
    """
    logger.info("Adding variants started!")
    if atac_mod not in mdata.mod_names:
        logger.error(f"Cannot find modality: {atac_mod}")
        return None
    adata = mdata[atac_mod]

    if variants_key not in adata.uns:
        logger.error(f"Cannot find variants in {atac_mod} modality")
        return None

    if sequence_var_key not in adata.var:
        logger.error(f"Cannot find sequence in {atac_mod} modality")
        return None

    df_peaks = adata.var.copy()
    df_variants = adata.uns[variants_key].copy()

    df_peaks["peak"] = df_peaks.index
    df_peaks = df_peaks.reset_index(drop=True)

    sample_list = adata.obs[sample_col_key].unique()

    # create dataframe for peaks and samples
    # assume that seq_1 is for chromatid 1 and seq_2 is for chromatid 2
    logger.info(
        f"Create dataframe for all {len(df_peaks)} peaks and {len(sample_list)} samples"
    )
    df_seq = pd.DataFrame(
        columns=["peak", "sample", "seq_1", "seq_2"],
        index=range(len(df_peaks) * len(sample_list)),
    )
    df_seq["peak"] = np.repeat(list(df_peaks["peak"]), len(sample_list))
    df_seq["sample"] = np.tile(sample_list, len(df_peaks))
    df_seq["start"] = np.repeat(list(df_peaks["start"]), len(sample_list))
    df_seq["seq_1"] = np.repeat(df_peaks[sequence_var_key].tolist(), len(sample_list))
    df_seq["seq_2"] = np.repeat(df_peaks[sequence_var_key].tolist(), len(sample_list))

    # update the sequences with variants
    # only update the sequences with heterozygous and homozygous alternate genotypes
    df_variants = df_variants[df_variants["genotype"].isin([1, 2])].reset_index(
        drop=True
    )

    logger.info(f"Number of variants with samples: {len(df_variants)}")
    if n_cpus == 1:
        df_seq = update_sequence_with_variants(df_seq, df_variants)
    else:
        logger.info(f"Using {n_cpus} CPUs for parallel processing.")
        from multiprocessing import Pool

        # split the df_variants by sample
        # and run the update_sequence_with_variants in parallel
        args = []
        for sample_id in sample_list:
            _df_seq = df_seq[df_seq["sample"] == sample_id].copy()
            _df_variants = df_variants[df_variants["sample"] == sample_id].copy()

            args.append((_df_seq, _df_variants))

        # run the update_sequence_with_variants in parallel
        with Pool(n_cpus) as pool:
            results = pool.starmap(
                update_sequence_with_variants,
                tqdm(args, desc="Updating sequences with variants"),
            )

        # combine the results
        df_seq = pd.concat(results, ignore_index=True)

    df_seq = df_seq.drop(columns=["start"])
    df_seq = df_seq.sort_values(by=["peak", "sample"])

    logger.info("Adding variants finished!")
    if inplace:
        adata.uns[seq_with_variants_key] = df_seq
        return None
    else:
        return df_seq
