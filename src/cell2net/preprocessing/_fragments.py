"""Functions to process scATAC-seq fragment file"""

import gzip
import os
import subprocess as sp

import numba
import numpy as np
import pandas as pd
import polars as pl
import pyBigWig

from cell2net._logging import logger
from cell2net.utils import santize_str_for_filename

from ._utils import bgzip, tabix_index


@numba.njit
def calculate_depth(
    chrom_size: int, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """
    Calculate genome depth for a given chromosome.

    This function computes the depth (coverage) at each base pair of a chromosome based on
    start and end positions of genomic fragments.

    Parameters
    ----------
    chrom_size :
        The size of the chromosome (total number of base pairs).
    starts :
        An array of start positions for the genomic fragments.
        Each value specifies the zero-based position where a fragment begins.
    ends :
        An array of end positions for the genomic fragments.
        Each value specifies the zero-based position where a fragment ends (exclusive).

    Returns
    -------
        A one-dimensional array of length `chrom_size`, where each position contains the
        depth (coverage) at that base pair.

    Notes
    -----
        - The `starts` and `ends` arrays must have the same length, as each pair defines a single fragment.
        - The depth is calculated as the count of overlapping fragments for each base pair.
        - This function uses Numba's Just-In-Time (JIT) compilation to optimize performance, making it suitable for processing large datasets.

    Examples
    --------
    >>> import numpy as np
    >>> import cell2net as cn
    >>> chrom_size = 10
    >>> starts = np.array([0, 2, 4])
    >>> ends = np.array([3, 6, 8])
    >>> depth = cn.pp.calculate_depth(chrom_size, starts, ends)
    >>> print(depth)
    array([1, 1, 2, 1, 2, 2, 1, 1, 0, 0], dtype=uint32)
    """
    # Initialize array for current chromosome to store the depth per basepair.
    chrom_depth = np.zeros(chrom_size, dtype=np.uint32)

    # Require same number of start and end positions.
    assert starts.shape[0] == ends.shape[0]

    for i in range(starts.shape[0]):
        # Add 1 depth for each basepair in the current fragment.
        chrom_depth[starts[i] : ends[i]] += numba.uint32(1)  # type: ignore

    return chrom_depth


@numba.njit
def collapse_consecutive_values(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse consecutive identical values in an array.

    This function identifies segments of consecutive identical values in an input
    array and returns their start indices, unique values, and lengths of each segment.

    Parameters
    ----------
    X :
        A 1D array of values (integers or floats) to process.

    Returns
    -------
        A tuple containing:

            - idx (numpy.ndarray): Start indices of each segment of consecutive identical values.
            - values (numpy.ndarray): The unique values corresponding to each segment.
            - lengths (numpy.ndarray): The lengths (number of repetitions) of each segment.

    Notes
    -----
        - This function is optimized for performance using `numba.prange` for parallel processing of the input array.
        - The output `idx` array contains the indices where each segment starts in `X`.
        - The `values` array contains the unique values from the input array, and the `lengths` array contains the counts of consecutive occurrences of each value.
        - To reconstruct the original input, use `values.repeat(lengths)`.

    Examples
    --------
    >>> import numpy as np
    >>> import cell2net as cn
    >>> X = np.array([1, 1, 2, 2, 2, 3, 1, 1])
    >>> idx, values, lengths = cn.pp.collapse_consecutive_values(X)
    >>> print(idx)
    ... [0 2 5 6]
    >>> print(values)
    ... [1. 2. 3. 1.]
    >>> print(lengths)
    ... [2 3 1 2]
    >>> np.array_equal(X, values.repeat(lengths))
    ... True
    """
    # Length.
    n = X.shape[0]

    # Create idx array with enough space to store all possible indices
    # in case there are no consecutive values in the input that are
    # the same.
    idx = np.empty(n + 1, dtype=np.uint32)

    # First index position will always be zero.
    idx[0] = 0
    # Next index postion to fill in in idx.
    j = 1

    # Loop over the whole input array and store indices for only those
    # positions for which the previous value was different.
    for i in numba.prange(1, n):
        if X[i - 1] == X[i]:
            continue

        # Current value is different from previous value, so store the index
        # position of the current index i in idx[j].
        idx[j] = i

        # Increase next index position to fill in in idx.
        j += 1

    # Store length of input as last value. Needed to calculate the number of times
    # the last consecutive value gets repeated.
    idx[j] = n

    # Get all consecutive different values from the input.
    values = X[idx[:j]].astype(np.float32)

    # Calculate the number of times each value in values gets consecutively
    # repeated in the input.
    lengths = idx[1 : j + 1] - idx[:j]

    # Restrict indices array to same number of element than the values and lentgths
    # arrays.
    idx = idx[:j].copy()

    # To reconstruct the original input: X == values.repeat(lenghts)
    return idx, values, lengths


def fragments_to_coverage(
    df_fragments: pl.DataFrame,
    chrom_sizes: dict[str, int],
    normalize: bool = True,
    scaling_factor: float = 1.0,
    cut_sites: bool = False,
    extend_cut_sites: int = 0,
):
    """
    Convert fragment data to genome coverage signal.

    This function processes fragment data and generates genome coverage or cut-site
    signal, which can be used for creating BigWig files or similar outputs.

    Parameters
    ----------
    df_fragments :
        A Polars DataFrame containing fragment data. Must include the columns:
        'Chromosome', 'Start', and 'End'.
    chrom_sizes:
        Dictionary mapping chromosome names to their respective sizes.
    normalize:
        If True, normalize the coverage values to Reads Per Million (RPM). Default is True.
    scaling_factor :
        A scaling factor to apply to the signal values. Only used if `normalize` is True.
        Default is 1.0.
    cut_sites:
        Use 1 bp Tn5 cut sites (start and end of each fragment) instead of whole
        fragment length for coverage calculation.
    extend_cut_sites:
        If set cut_sites, expand cut sites for both upstream and downstream, by default: 0

    Yields
    ------
    A tuple containing:

            - chroms (numpy.ndarray): Chromosome names for each coverage interval.
            - starts (numpy.ndarray): Start positions of coverage intervals.
            - ends (numpy.ndarray): End positions of coverage intervals.
            - values (numpy.ndarray): Signal values for each coverage interval.

    Notes
    -----
        - The `df_fragments` DataFrame is partitioned by chromosome for efficient processing.
        - The `chrom_sizes` dictionary defines the size of each chromosome and is used to initialize arrays.
        - If `cut_sites` is True, the coverage is computed at the fragment boundaries rather than the entire fragment range.
        - Normalization scales the signal to RPM, and an additional scaling factor can further adjust the signal values.

    Examples
    --------
    >>> import polars as pl
    >>> import cell2net as cn
    >>> df_fragments = pl.DataFrame({
    ...     "Chromosome": ["chr1", "chr1", "chr2"],
    ...     "Start": [100, 200, 300],
    ...     "End": [150, 250, 350]
    ... })
    >>> chrom_sizes = {"chr1": 1000, "chr2": 500}
    >>> results = cn.pp.fragments_to_coverage(df_fragments, chrom_sizes, normalize=False)
    >>> for chroms, starts, ends, values in results:
    ...     print(chroms, starts, ends, values)
    """
    chrom_arrays = {}

    for chrom, chrom_size in chrom_sizes.items():
        chrom_arrays[chrom] = np.zeros(chrom_size, dtype=np.uint32)

    n_fragments = 0

    logger.info("Split fragments by chromosome")
    per_chrom_fragments_dfs = {
        str(chrom): fragments_chrom_df_pl
        for (chrom,), fragments_chrom_df_pl in df_fragments.partition_by(
            ["Chromosome"],
            as_dict=True,
        ).items()
    }

    logger.info("Calculate depth per chromosome:")
    for chrom in per_chrom_fragments_dfs:
        if chrom not in chrom_sizes:
            logger.warning(f"Skipping {chrom} as it is not in chrom sizes file.")
            continue

        starts, ends = (
            per_chrom_fragments_dfs[chrom].select(["Start", "End"]).to_numpy().T
        )

        if cut_sites:
            # Create cut site positions (for both start and end of a fragment).
            # starts, ends = (
            #     np.hstack((starts, ends - 1)),
            #     np.hstack((starts + 1, ends)),
            # )
            starts, ends = (
                np.hstack((starts - extend_cut_sites, ends - extend_cut_sites - 1)),
                np.hstack((starts + extend_cut_sites + 1, ends + extend_cut_sites)),
            )

        chrom_arrays[chrom] = calculate_depth(chrom_sizes[chrom], starts, ends)
        n_fragments += per_chrom_fragments_dfs[chrom].height

    # Calculate RPM scaling factor.
    rpm_scaling_factor = n_fragments / 1_000_000.0
    logger.info(
        "Compact depth array per chromosome (make ranges for consecutive the same values and remove zeros):"
    )
    for chrom in chrom_sizes:
        idx, values, lengths = collapse_consecutive_values(chrom_arrays[chrom])
        non_zero_idx = np.flatnonzero(values)

        if non_zero_idx.shape[0] == 0:
            # Skip chromosomes with no depth > 0.
            continue

        # Select only consecutive different values and calculate start and end
        # coordinates (in BED format) for each of those ranges.
        chroms = np.repeat(chrom, len(non_zero_idx))
        starts = idx[non_zero_idx]
        ends = idx[non_zero_idx] + lengths[non_zero_idx]
        values = values[non_zero_idx]

        if normalize:
            values = values / rpm_scaling_factor * scaling_factor
        elif scaling_factor != 1.0:
            values *= scaling_factor

        yield chroms, starts, ends, values


def fragment_to_bigwig(
    fragment_file: str,
    chrom_sizes: dict[str, int],
    bw_filename: str,
    normalize: bool = True,
    scaling_factor: float = 1.0,
    cut_sites: bool = False,
    extend_cut_sites: int = 0,
    cell_barcodes: list[str] | None = None,
) -> None:
    """
    Convert fragment file to BigWig format.

    This function reads a fragment file, calculates coverage or cut-site signal,
    and writes the resulting data to a BigWig file.

    Parameters
    ----------
    fragment_file :
        Path to the input fragment file.
        The file can be plain text or gzip-compressed (".gz").
    chrom_sizes :
        A dictionary of chromosome sizes, e.g., {"chr1": 248956422, "chr2": 242193529, ...}.
    bw_filename :
        Path to the output BigWig file.
    normalize :
        If True, normalize coverage or signal values. Default is True.
    scaling_factor : float, optional
        Factor to scale signal values if `normalize` is True. Default is 1.0.
    cut_sites :
        If True, compute the cut-site signal instead of coverage. Default is False.
    extend_cut_sites:
        If set cut_sites, expand cut sites for both upstream and downstream, by default: 0

    Returns
    -------
    Write output to bigwig file

    Notes
    -----
        - The input fragment file should be tab-delimited and follow the format: Chromosome, Start, End, Barcode, Count.
        - Lines starting with `#` or empty lines are skipped during parsing.
        - Uses `pyBigWig` for writing BigWig files and `polars` for efficient data manipulation.

    Example
    -------
    >>> fragment_file = "example_fragments.tsv.gz"
    >>> chrom_sizes = {"chr1": 248956422, "chr2": 242193529}
    >>> bw_filename = "output.bw"
    >>> fragment_to_bigwig(fragment_file, chrom_sizes, bw_filename, normalize=True, scaling_factor=1.0, cut_sites=False)
    """
    open_fn = gzip.open if fragment_file.endswith(".gz") else open
    skip_rows = 0
    with open_fn(fragment_file, "rt") as f:
        for line in f:
            line = line.strip()
            # Count number of empty lines and lines which start with a comment
            # before the actual data.
            if not line or line.startswith("#"):
                skip_rows += 1
            else:
                break

    logger.info(f"Reading fragments from {fragment_file}")
    df_fragments = pl.read_csv(
        fragment_file,
        skip_rows=skip_rows,
        has_header=False,
        separator="\t",
        use_pyarrow=False,
        new_columns=["Chromosome", "Start", "End", "Barcode", "Count"],
    )

    # filter out cell barcodes if provided
    if cell_barcodes is not None:
        logger.info("Filtering fragments by cell barcodes")
        df_fragments = df_fragments.filter(pl.col("Barcode").is_in(cell_barcodes))

    logger.info(f"Number of fragments: {df_fragments.height}")

    with pyBigWig.open(bw_filename, "wb") as bw:
        logger.info("Add chromosome sizes to bigwig header")
        bw.addHeader(list(chrom_sizes.items()))

        fragments_to_coverage_chrom_iter = fragments_to_coverage(
            df_fragments=df_fragments,
            chrom_sizes=chrom_sizes,
            normalize=normalize,
            scaling_factor=scaling_factor,
            cut_sites=cut_sites,
            extend_cut_sites=extend_cut_sites,
        )

        for chroms, starts, ends, values in fragments_to_coverage_chrom_iter:
            bw.addEntries(chroms=chroms, starts=starts, ends=ends, values=values)

    return None


def split_fragments(
    fragment_files: str | list[str],
    cell_barcodes: list[str],
    groups: list[str],
    out_dir: str,
) -> None:
    """
    Splits a fragment file into multiple group-specific fragment files based on cell barcodes.

    This function reads a fragment file, assigns each fragment to a group based on the cell barcode,
    and writes group-specific fragments into separate files.
    The output files are compressed and indexed using bgzip and tabix.

    Parameters
    ----------
    fragment_files :
        Path to the input fragment files.
        Each file can be a plain text or gzip-compressed (.gz) file and
        should have the following formats:

        +-----+-------+--------+----------------------+-----+
        |chr1 | 10012 |  10013 |   TTTGCGACACCCACAG-1 |   1 |
        +-----+-------+--------+----------------------+-----+
        |chr1 | 10066 |  10198 |   ACGAATCTCATTTGCT-1 |   1 |
        +-----+-------+--------+----------------------+-----+
        |chr1 | 10066 |  10478 |   TCAAGAACAGTAATAG-1 |   1 |
        +-----+-------+--------+----------------------+-----+

    cell_barcodes:
        A list of cell barcodes corresponding to the fragments.
    groups:
        A list of group names corresponding to each cell barcode.
        This can represent cell types or states, or different conditions.
        Must have the same length as `cell_barcodes`.
    out_dir:
        Path to the output directory where the group-specific fragment files will be saved.

    Returns
    -------
    Write output to fragment file

    Notes
    -----
        - For each unique group in `groups`, a compressed and indexed fragment file is created in the output directory.
        - The files are named as `<group>.fragments.tsv.gz`.
    """
    # check if the barcodes and groups have same length
    assert len(cell_barcodes) == len(
        groups
    ), "Cell barcodes and groups have different length"

    # make group name safe for use as a filename
    groups = [santize_str_for_filename(s) for s in groups]

    group_barcode_dict = pd.Series(groups, index=cell_barcodes).to_dict()

    # create a files to write fragments
    logger.info("Create output files")
    file_handles = {}
    for group in set(groups):
        file_name = f"{out_dir}/{group}.fragments.unsort.tsv"
        file_handles[group] = open(file_name, "w")

    logger.info("Split fragments by groups")
    if isinstance(fragment_files, str):
        fragment_files = [fragment_files]

    for fragment_file in fragment_files:
        open_fn = gzip.open if fragment_file.endswith(".gz") else open
        with open_fn(fragment_file, "rt") as f:
            for line in f:
                # Remove newlines and spaces.
                line = line.strip()

                # Skip lines with #
                if not line or line.startswith("#"):
                    continue

                # Assuming the 4th column is the cell barcode
                columns = line.strip().split("\t")
                cell_barcode = columns[3]

                # Get the corresponding cell type and write to the respective file
                if cell_barcode in group_barcode_dict:
                    cell_type = group_barcode_dict[cell_barcode]
                    file_handles[cell_type].write(line + "\n")

    # Close output files
    for group in set(groups):
        file_handles[group].close()

    # sort, compress and index the fragment file
    logger.info("Sort, compress and index the fragment files")
    for group in set(groups):
        sp.run(
            f"sort -k1,1 -k2,2n {out_dir}/{group}.fragments.unsort.tsv > {out_dir}/{group}.fragments.tsv",
            shell=True,
            check=True,
        )
        bgzip(filename=f"{out_dir}/{group}.fragments.tsv")
        tabix_index(
            filename=f"{out_dir}/{group}.fragments.tsv.gz",
            preset="bed",
            chrom=1,
            start=2,
            end=3,
        )
        os.remove(f"{out_dir}/{group}.fragments.unsort.tsv")

    return None
