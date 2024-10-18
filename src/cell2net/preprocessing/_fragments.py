"""Functions to process scATAC-seq fragment file"""

import gzip

import pyBigWig  # type: ignore


def read_fragments(fragment_file: str):

    open_fn = gzip.open if fragment_file.endswith(".gz") else open

    return NotImplemented


def fragment_to_bigwig(
    fragment_file: str,
    chrom_sizes: dict[str, int],
    out_bigwig: str,
    barcodes: list[str] | None = None,
    normalize: bool = True,
    scaling_factor: int = 1_000_000,
):
    """
    Convert a fragment file to a bigwig file.

    Parameters
    ----------
    fragment_file : str
        Path to the fragment file (e.g., fragment.tsv.gz).
    chrom_sizes : dict[str, int]
        A dictionary of chromosome sizes, e.g., {"chr1": 248956422, "chr2": 242193529, ...}.
    out_bigwig : str
        File name of the output bigWig file.
    barcodes: list[str]
        A list of barcodes used to filter the cells. If None, will use all cells. Default: None
    normalize : bool, optional
        Whether or not normalize the signal. Default: True
    scaling_factor : int, optional
        Scaling factor for normalization. Default: 1000000

    Returns
    -------
    _type_
        _description_
    """
    # Open a bigWig file for writing
    bw = pyBigWig.open(out_bigwig, "w")

    # Define chromosome sizes in the bigWig file
    bw.addHeader(list(chrom_sizes.items()))

    return NotImplemented


def split_fragment(
    fragmet_file: str,
    groups: list[str],
    chromsizes: dict[str, int],
    out_dir: str,
) -> None:
    """
    Split the input fragment file using the groups

    Parameters
    ----------
    fragmet_file : str
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
    return None
