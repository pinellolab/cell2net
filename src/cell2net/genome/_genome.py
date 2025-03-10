# modified from https://github.com/kaizhang/SnapATAC2/blob/main/snapatac2-python/python/snapatac2/genome.py
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pooch import Decompress

from cell2net.datasets import register_datasets


class Genome:
    """
    A class that encapsulates a genome, including its sequence (FASTA), annotation, and chromosome sizes.

    This class provides convenient access to genome-related data, such as the FASTA file,
    annotation file, and chromosome sizes. Chromosome sizes are calculated automatically
    if not provided during initialization.

    Parameters
    ----------
    name :
        Name of the genome (e.g., "hg38", "mm10").

    fasta :
        Path to the FASTA file containing the genome sequence.

    annotation :
        Path to the annotation file (e.g., GTF or GFF3).

    chrom_sizes :
        A dictionary with chromosome names as keys and their lengths as values.
        If not provided, chromosome sizes will be computed from the FASTA file.

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

    Methods
    -------
    fasta
        Returns the path to the FASTA file.

    annotation
        Returns the path to the annotation file.

    chrom_sizes
        Returns the chromosome sizes as a dictionary.
        If not provided during initialization, they are computed from the FASTA file.
    """

    def __init__(
        self,
        name: str,
        fasta: Path | Callable[[], Path],
        annotation: Path | Callable[[], Path],
        chrom_sizes: dict[str, int] | None = None,
    ) -> None:
        self._name = name

        if callable(fasta):
            self._fetch_fasta = fasta
            self._fasta = None
        elif isinstance(fasta, Path) or isinstance(fasta, str):
            self._fasta = Path(fasta)
            self._fetch_fasta = None
        else:
            raise ValueError("fasta must be a Path or Callable")

        if callable(annotation):
            self._fetch_annotation = annotation
            self._annotation = None
        elif isinstance(annotation, Path) or isinstance(annotation, str):
            self._annotation = Path(annotation)
            self._fetch_annotation = None
        else:
            raise ValueError("annotation must be a Path or Callable")

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
        if self._fasta is None:
            self._fasta = Path(self._fetch_fasta())  # type: ignore
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
        if self._annotation is None:
            self._annotation = Path(self._fetch_annotation())  # type: ignore
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
            from pyfaidx import Fasta

            fasta = Fasta(self.fasta)
            self._chrom_sizes = {chr: len(fasta[chr]) for chr in fasta.keys()}
        return self._chrom_sizes


hg19 = Genome(
    name="hg19",
    fasta=lambda: register_datasets().fetch(  # type: ignore
        "gencode_v41_GRCh37.fa.gz",
        processor=Decompress(method="gzip"),
        progressbar=True,
    ),
    annotation=lambda: register_datasets().fetch(  # type: ignore
        "gencode_v41_GRCh37.gff3.gz", progressbar=True
    ),
)

hg38 = Genome(
    name="hg38",
    fasta=lambda: register_datasets().fetch(  # type: ignore
        "gencode_v41_GRCh38.fa.gz",
        processor=Decompress(method="gzip"),
        progressbar=True,
    ),
    annotation=lambda: register_datasets().fetch(  # type: ignore
        "gencode_v41_GRCh38.gff3.gz", progressbar=True
    ),
    chrom_sizes={
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
        "chrM": 16569,
    },
)

mm39 = Genome(
    name="mm39",
    fasta=lambda: register_datasets().fetch(  # type: ignore
        "gencode_vM30_GRCm39.fa.gz",
        processor=Decompress(method="gzip"),
        progressbar=True,
    ),
    annotation=lambda: register_datasets().fetch(  # type: ignore
        "gencode_vM30_GRCm39.gff3.gz", progressbar=True
    ),
    chrom_sizes={
        "chr1": 195154279,
        "chr2": 181755017,
        "chr3": 159745316,
        "chr4": 156860686,
        "chr5": 151758149,
        "chr6": 149588044,
        "chr7": 144995196,
        "chr8": 130127694,
        "chr9": 124359700,
        "chr10": 130530862,
        "chr11": 121973369,
        "chr12": 120092757,
        "chr13": 120883175,
        "chr14": 125139656,
        "chr15": 104073951,
        "chr16": 98008968,
        "chr17": 95294699,
        "chr18": 90720763,
        "chr19": 61420004,
        "chrX": 169476592,
        "chrY": 91455967,
        "chrM": 16299,
    },
)  # type: ignore

mm10 = Genome(
    name="mm10",
    fasta=lambda: register_datasets().fetch(  # type: ignore
        "gencode_vM25_GRCm38.fa.gz",
        processor=Decompress(method="gzip"),
        progressbar=True,
    ),
    annotation=lambda: register_datasets().fetch(  # type: ignore
        "gencode_vM25_GRCm38.gff3.gz", progressbar=True
    ),
    chrom_sizes={
        "chr1": 195471971,
        "chr2": 182113224,
        "chr3": 160039680,
        "chr4": 156508116,
        "chr5": 151834684,
        "chr6": 149736546,
        "chr7": 145441459,
        "chr8": 129401213,
        "chr9": 124595110,
        "chr10": 130694993,
        "chr11": 122082543,
        "chr12": 120129022,
        "chr13": 120421639,
        "chr14": 124902244,
        "chr15": 104043685,
        "chr16": 98207768,
        "chr17": 94987271,
        "chr18": 90702639,
        "chr19": 61431566,
        "chrX": 171031299,
        "chrY": 91744698,
        "chrM": 16299,
    },
)  # type: ignore
