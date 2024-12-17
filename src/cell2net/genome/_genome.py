# modified from https://github.com/kaizhang/SnapATAC2/blob/main/snapatac2-python/python/snapatac2/genome.py
from __future__ import annotations

from pathlib import Path

from pyfaidx import Fasta


class Genome:
    """
    A class that encapsulates information about a genome, including its FASTA sequence, its annotation, and chromosome sizes.

    Attributes
    ----------
    name: str
        Genome name
    fasta
        The path to the FASTA file.
    annotation
        The path to the annotation file.
    chrom_sizes
        A dictionary containing chromosome names and sizes.

    """

    def __init__(
        self,
        name: str,
        fasta: Path | str,
        annotation: Path | str,
        chrom_sizes: dict[str, int] | None = None,
    ) -> None:
        self._name = name
        self._fasta = fasta
        self._annotation = annotation
        self._chrom_sizes = chrom_sizes

    @property
    def fasta(self):
        """
        The Path to the FASTA file.

        Returns
        -------
        Path
            The path to the FASTA file.
        """
        return self._fasta

    @property
    def annotation(self):
        """
        The Path to the annotation file.

        Returns
        -------
        Path
            The path to the annotation file.
        """
        return self._annotation

    @property
    def chrom_sizes(self):
        """
        A dictionary with chromosome names as keys and their lengths as values.

        Returns
        -------
        dict[str, int]
            A dictionary of chromosome sizes.
        """
        if self._chrom_sizes is None:
            fasta = Fasta(self.fasta)
            self._chrom_sizes = {chr: len(fasta[chr]) for chr in fasta.keys()}
        return self._chrom_sizes


# GRCh37 = Genome(
#     fasta=lambda: register_datasets().fetch(
#         "gencode_v41_GRCh37.fa.gz",
#         processor=Decompress(method="gzip"),
#         progressbar=True,
#     ),
#     annotation=lambda: register_datasets().fetch(
#         "gencode_v41_GRCh37.gff3.gz", progressbar=True
#     ),
# )
# hg19 = GRCh37

hg38 = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
}


def get_chrom_sizes(genome: str) -> dict:
    return hg38
