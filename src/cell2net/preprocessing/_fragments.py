"""Functions to process scATAC-seq fragment file"""

import gzip
import subprocess

import numba
import numpy as np
import pandas as pd
import polars as pl
import pyBigWig

from cell2net._logging import logger


@numba.njit
def calculate_depth(chrom_size, starts, ends):
    """Calculate depth per basepair for a chromosome based on starts end ends of fragments on the current chromosome."""
    # Initialize array for current chromosome to store the depth per basepair.
    chrom_depth = np.zeros(chrom_size, dtype=np.uint32)

    # Require same number of start and end positions.
    assert starts.shape[0] == ends.shape[0]

    for i in range(starts.shape[0]):
        # Add 1 depth for each basepair in the current fragment.
        chrom_depth[starts[i] : ends[i]] += numba.uint32(1)  # type: ignore

    return chrom_depth


@numba.njit
def collapse_consecutive_values(X):
    """Collapse consecutive values in array and return indices, values and lengths."""
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
):
    """
    Calculate genome coverage for fragments and yield per chromosome a chroms, starts, ends and values numpy array.

    Parameters
    ----------
    df_fragments
        Polars DataFrame with fragments.
    chrom_sizes
        Dictionary with chromosome names as keys and chromosome sizes as values.
    normalize
        Whether to normalize the coverage by dividing by the number of fragments
        multiplied by 1 million.
    scaling_factor
        Scaling factor for coverage data. If normalization is enabled, scaling is
        applied afterwards.
    cut_sites
        Use 1 bp Tn5 cut sites (start and end of each fragment) instead of whole
        fragment length for coverage calculation.
    verbose
        Whether to print progress.

    """
    chrom_arrays = {}

    for chrom, chrom_size in chrom_sizes.items():
        chrom_arrays[chrom] = np.zeros(chrom_size, dtype=np.uint32)

    n_fragments = 0

    logger.info("Split fragments df by chromosome")
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
            logger.warning(f"   Skipping {chrom} as it is not in chrom sizes file.")
            continue

        starts, ends = (
            per_chrom_fragments_dfs[chrom].select(["Start", "End"]).to_numpy().T
        )

        if cut_sites:
            # Create cut site positions (for both start and end of a fragment).
            starts, ends = (
                np.hstack((starts, ends - 1)),
                np.hstack((starts + 1, ends)),
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
) -> None:
    """
    Calculate genome coverage for fragments and write to a bigWig file.

    Parameters
    ----------
    fragment_file : str
        Path to the fragment file (e.g., fragment.tsv.gz).
    chrom_sizes : dict[str, int]
        A dictionary of chromosome sizes, e.g., {"chr1": 248956422, "chr2": 242193529, ...}.
    bw_filename : str
        File name of the output bigWig file.
    barcodes: list[str]
        A list of barcodes used to filter the cells. If None, will use all cells. Default: None
    normalize : bool, optional
        Whether or not normalize the signal. Default: True
    scaling_factor : int, optional
        Scaling factor for normalization. Default: 1000000

    Returns
    -------
    None
        Write output to bigwig file
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
        )

        logger.info("Add signal bigwig")
        for chroms, starts, ends, values in fragments_to_coverage_chrom_iter:
            bw.addEntries(chroms=chroms, starts=starts, ends=ends, values=values)

    return None


def split_fragment(
    fragment_file: str,
    cell_barcodes: list[str],
    groups: list[str],
    out_dir: str,
) -> None:
    """
    Split the input fragment file using the groups

    Parameters
    ----------
    fragment_file : str
        File name of the fragment file, optionally compressed with gzip or zstd.
        A fragment should have at least 4 columns:
        chromosome, start, end, barcode.
        Optionally, it can have additional column indicating the count of barcode:
        chromosome, start, end, barcode, count
    groups : list[str]
        A list of strings defining the group of each barcode.
        This can refer to cell types or states, or different conditions.
    out_dir : str
        Output directory

    Returns
    -------
    None
        _description_
    """
    group_barcode_dict = pd.Series(groups, index=cell_barcodes).to_dict()

    # create a files to write fragments
    logger.info("Create output files")
    file_handles = {}
    for group in set(groups):
        file_name = f"{out_dir}/{group}.fragments.tsv"
        file_handles[group] = open(file_name, "w")

    logger.info("Split fragments by groups")
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

    # compress and index the fragment file using bgzip and tabix
    logger.info("Compress and index fragment files")
    for group in set(groups):
        subprocess.run(["bgzip", f"{out_dir}/{group}.fragments.tsv"])
        subprocess.run(
            ["tabix", "-s1", "-b2", "-e3", f"{out_dir}/{group}.fragments.tsv.gz"]
        )

    return None
