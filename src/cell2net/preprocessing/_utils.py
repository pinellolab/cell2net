import subprocess
from typing import Literal


def bgzip(filename: str):
    """
    Call bgzip to compress a file

    Parameters
    ----------
    filename : str
        Input filename
    """
    subprocess.run(["bgzip", "-f", filename])


def tabix_index(
    filename: str,
    preset: Literal["gff", "bed", "sam", "vcf", "gaf"] = "bed",
    chrom: int = 1,
    start: int = 2,
    end: int = 3,
    skip: int = 0,
    comment: str = "#",
):
    """
    Call tabix to create an index for a bgzip-compressed file

    Parameters
    ----------
    filename : str
        Name of a bgzip-compressed file
    preset : str, optional
        File format, by default "bed"
    chrom : int, optional
        Column number for sequence names, by default 1
    start : int, optional
        Column number for region start, by default 2
    end : int, optional
        Column number for region end, by default 3
    skip : int, optional
        Numer of lines to skip first, by default 0
    comment : str, optional
        Skip comment lines starting with, by default "#"
    """
    subprocess.run(
        [
            "tabix",
            "-p",
            preset,
            "-s",
            str(chrom),
            "-b",
            str(start),
            "-e",
            str(end),
            "-S",
            str(skip),
            "-c",
            comment,
            filename,
        ]
    )
