"""Functions to process DNA sequences"""

import numpy as np
from mudata import MuData
from pysam import FastaFile
from tqdm import tqdm

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
        "N": [0, 0, 0, 0],
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


def add_variants_to_sequence(
    mdata: MuData,
    ref_fasta: str,
    atac_mod: str = "atac",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    variants_key: str = "variants",
) -> None:
    pass


# def add_variants_to_sequence(
#     mdata: MuData,
#     ref_fasta: str,
#     atac_mod: str = "atac",
#     chr_var_key: str = "chr",
#     start_var_key: str = "start",
#     end_var_key: str = "end",
#     variants_key: str = "variants",
# ) -> None:
#     """
#     Add sequences to peak metadata in a MuData object by considering genomic variants.

#     This function retrieves DNA sequences for genomic regions specified in the `.var`
#     attribute of the AnnData object within a MuData object. The sequences are fetched
#     from a reference FASTA file and added as metadata under the specified key.

#     Parameters
#     ----------
#     mdata :
#         A MuData object containing the modality with peak metadata.
#     ref_fasta :
#         Path to the reference FASTA file. This file must be indexed (e.g., with samtools faidx).
#     mod_name :
#         The name of the modality containing peak data. Defaults to "atac".
#     chr_var_key :
#         The key in `.var` that contains chromosome names. Defaults to "chr".
#     start_var_key :
#         The key in `.var` that contains the start positions of peaks. Defaults to "start".
#     end_var_key :
#         The key in `.var` that contains the end positions of peaks. Defaults to "end".
#     sequence_var_key :
#         The key under which the retrieved DNA sequences will be stored in `.var`. Defaults to "dna_sequence".

#     Returns
#     -------
#     None
#         The function modifies the MuData object in place by adding DNA sequences to the
#         specified key in the `.var` attribute.

#     Raises
#     ------
#     AssertionError
#         If the specified modality (`mod_name`) is not found in the MuData object.
#     FileNotFoundError
#         If the `ref_fasta` file does not exist or is not properly indexed.

#     Examples
#     --------
#     >>> from mudata import MuData
#     >>> import anndata as ad
#     >>> import pandas as pd
#     >>> import cell2net as cn
#     >>> data = ad.AnnData(var=pd.DataFrame({
#     ...     "chr": ["chr1", "chr2"],
#     ...     "start": [100, 200],
#     ...     "end": [150, 250]
#     ... }))
#     >>> mdata = MuData({"atac": data})
#     >>> cn.pp.add_dna_sequence(mdata, ref_fasta="reference.fasta")
#     >>> print(mdata["atac"].var["dna_sequence"])
#     0    ATCGTTGAC...
#     1    TGGCCAATA...
#     """
#     if atac_mod not in mdata.mod_names:
#         logger.error(f"Cannot find modality: {atac_mod}")
#         return None

#     adata = mdata[atac_mod]

#     if variants_key not in adata.uns:
#         logger.error(f"Cannot find variants in {atac_mod} modality")
#         return None

#     df_peaks = adata.var[[chr_var_key, start_var_key, end_var_key]]
#     df_var = adata.uns[variants_key]

#     # for test
#     df_peaks = df_peaks.head(1000)

#     fasta = FastaFile(filename=ref_fasta)

#     # we get the sequences from sister chromatids
#     df_seq1 = pd.DataFrame(columns=sample_ids, index=df_peaks.index)
#     df_seq2 = pd.DataFrame(columns=sample_ids, index=df_peaks.index)
#     for chrom, start, end in tqdm(
#         zip(
#             df_peaks[chr_var_key],
#             df_peaks[start_var_key],
#             df_peaks[end_var_key],
#             strict=False,
#         )
#     ):
#         # get reference sequence
#         ref_seq = fasta.fetch(chrom, start, end).upper()

#         df_var = get_genomic_variants(reader, chrom, start, end)

#         if len(df_var) == 0:
#             # no variants in this region, use reference sequence for both seqs
#             df_seq1.loc[df_peaks.index, sample_ids] = ref_seq
#             df_seq2.loc[df_peaks.index, sample_ids] = ref_seq
#         else:
#             # get the sequence with variants for each sample
#             for _, sample_id in enumerate(sample_ids):
#                 _df_var = df_var[df_var["sample"] == sample_id]

#                 # initialize the sequences with the reference sequence
#                 # for both seq1 and seq2
#                 # we assume that seq1 is the reference sequence
#                 # and seq2 is the alternate sequence
#                 seq1 = seq2 = ref_seq

#                 for _, row in _df_var.iterrows():
#                     if row["genotype"] == 0 or row["genotype"] == np.nan:
#                         # homozygous reference or missing genotype information
#                         continue
#                     elif row["genotype"] == 1:
#                         # heterozygous
#                         seq2 = (
#                             seq2[: row["pos"] - start]
#                             + row["alt"]
#                             + seq2[row["pos"] - start + 1 :]
#                         )
#                     elif row["genotype"] == 2:
#                         # homozygous alternate
#                         seq1 = (
#                             seq1[: row["pos"] - start]
#                             + row["alt"]
#                             + seq1[row["pos"] - start + 1 :]
#                         )
#                         seq2 = (
#                             seq2[: row["pos"] - start]
#                             + row["alt"]
#                             + seq2[row["pos"] - start + 1 :]
#                         )

#                 # update the sequence in the dataframe
#                 df_seq1.loc[df_peaks.index, sample_id] = seq1
#                 df_seq2.loc[df_peaks.index, sample_id] = seq2

#     adata.uns["dna_sequence_1"] = df_seq1
#     adata.uns["dna_sequence_2"] = df_seq2

#     return None
