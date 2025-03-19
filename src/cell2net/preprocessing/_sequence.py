"""Functions to process DNA sequences"""

import numpy as np
import torch
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


def seq_to_one_hot(seq: str) -> torch.Tensor:
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
    one_hot = torch.tensor([nuc_d[x] for x in seq], dtype=torch.float32)

    return one_hot


def one_hot_to_seq(one_hot: torch.Tensor) -> str:
    """
    Convert a one-hot encoded DNA sequence to a string.

    Parameters
    ----------
    one_hot : torch.Tensor
        A one-hot encoded DNA sequence with shape (sequence_length, 4). The
        tensor should be of type torch.float32. The sequence length is the
        first dimension of the tensor.

    Returns
    -------
    str
        The DNA sequence represented by the one-hot encoding.

    Examples
    --------
    >>> import torch
    >>> one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    >>> one_hot_to_str(one_hot)
    'ACGT'
    """
    # Dictionary returning nucleotide for each one-hot encoding
    mapping = {0: "A", 1: "C", 2: "G", 3: "T"}

    idxs = one_hot.argmax(axis=1)  # type: ignore

    # Handle unknown bases (if all zeros in one-hot row)
    sequence = "".join(
        mapping[i] if one_hot[row].sum() == 1 else "N" for row, i in enumerate(idxs)
    )

    return sequence


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
        )
    ):
        seqs.append(fasta.fetch(chrom, start, end).upper())

    adata.var[sequence_var_key] = seqs

    return None
