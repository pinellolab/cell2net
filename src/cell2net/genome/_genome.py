# modified from https://github.com/kaizhang/SnapATAC2/blob/main/snapatac2-python/python/snapatac2/genome.py
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Union

from pooch import Decompress

from cell2net.datasets import register_datasets

logger = logging.getLogger(__name__)

# Type aliases for better code readability
PathLike = Union[str, Path]
PathOrCallable = Union[PathLike, Callable[[], PathLike]]
ChromSizes = dict[str, int]


class GenomeError(Exception):
    """Base exception for genome-related errors."""
    pass


class GenomeFileNotFoundError(GenomeError):
    """Raised when genome files cannot be found or accessed."""
    pass


class GenomeValidationError(GenomeError):
    """Raised when genome validation fails."""
    pass


class Genome:
    """
    A class that encapsulates a genome, including its sequence (FASTA), annotation, and chromosome sizes.

    This class provides convenient access to genome-related data, such as the FASTA file,
    annotation file, and chromosome sizes. Chromosome sizes are calculated automatically
    if not provided during initialization.

    Parameters
    ----------
    name : str
        Name of the genome (e.g., "hg38", "mm10").
    fasta : PathLike or Callable[[], PathLike]
        Path to the FASTA file containing the genome sequence, or a callable that returns the path.
    annotation : PathLike or Callable[[], PathLike]
        Path to the annotation file (e.g., GTF or GFF3), or a callable that returns the path.
    chrom_sizes : dict[str, int], optional
        A dictionary with chromosome names as keys and their lengths as values.
        If not provided, chromosome sizes will be computed from the FASTA file.

    Attributes
    ----------
    name : str
        Genome name

    Raises
    ------
    GenomeError
        If genome initialization fails.
    GenomeFileNotFoundError
        If required genome files are not found.
    GenomeValidationError
        If genome validation fails.

    Examples
    --------
    Create a genome from local files:

    >>> genome = Genome(
    ...     name="custom_genome",
    ...     fasta="/path/to/genome.fa",
    ...     annotation="/path/to/annotation.gtf"
    ... )

    Create a genome with lazy-loaded files:

    >>> genome = Genome(
    ...     name="hg38",
    ...     fasta=lambda: download_hg38_fasta(),
    ...     annotation=lambda: download_hg38_annotation()
    ... )
    """

    # Class-level registry for available genomes
    _registry: dict[str, 'Genome'] = {}

    def __init__(
        self,
        name: str,
        fasta: PathOrCallable,
        annotation: PathOrCallable,
        chrom_sizes: Optional[ChromSizes] = None,
    ) -> None:
        self._name = self._validate_name(name)
        self._fasta_source = self._validate_path_or_callable(fasta, "fasta")
        self._annotation_source = self._validate_path_or_callable(annotation, "annotation")
        self._chrom_sizes = chrom_sizes

        # Cached properties
        self._fasta: Optional[Path] = None
        self._annotation: Optional[Path] = None
        self._computed_chrom_sizes: Optional[ChromSizes] = None

        # Register this genome instance
        self._registry[name] = self

    @staticmethod
    def _validate_name(name: str) -> str:
        """Validate genome name."""
        if not isinstance(name, str) or not name.strip():
            raise GenomeValidationError("Genome name must be a non-empty string")
        return name.strip()

    @staticmethod
    def _validate_path_or_callable(source: PathOrCallable, source_type: str) -> PathOrCallable:
        """Validate that source is either a valid path or callable."""
        if callable(source):
            return source
        elif isinstance(source, (str, Path)):
            return Path(source)
        else:
            raise GenomeValidationError(
                f"{source_type} must be a Path, string, or Callable, got {type(source)}"
            )

    def _resolve_path(self, source: PathOrCallable, source_type: str) -> Path:
        """Resolve a path source to an actual Path object."""
        try:
            if callable(source):
                resolved = source()
            else:
                resolved = source

            path = Path(resolved)
            if not path.exists():
                raise GenomeFileNotFoundError(f"{source_type} file not found: {path}")

            return path
        except Exception as e:
            raise GenomeFileNotFoundError(f"Failed to resolve {source_type} path: {e}") from e

    @property
    def name(self) -> str:
        """The genome name."""
        return self._name

    @property
    def fasta(self) -> Path:
        """
        The Path to the FASTA file.

        Returns
        -------
        Path
            The path to the FASTA file.

        Raises
        ------
        GenomeFileNotFoundError
            If the FASTA file cannot be found or accessed.
        """
        if self._fasta is None:
            self._fasta = self._resolve_path(self._fasta_source, "FASTA")
            logger.info(f"Resolved FASTA file for {self.name}: {self._fasta}")
        return self._fasta

    @property
    def annotation(self) -> Path:
        """
        The Path to the annotation file.

        Returns
        -------
        Path
            The path to the annotation file.

        Raises
        ------
        GenomeFileNotFoundError
            If the annotation file cannot be found or accessed.
        """
        if self._annotation is None:
            self._annotation = self._resolve_path(self._annotation_source, "annotation")
            logger.info(f"Resolved annotation file for {self.name}: {self._annotation}")
        return self._annotation

    @property
    def chrom_sizes(self) -> ChromSizes:
        """
        A dictionary with chromosome names as keys and their lengths as values.

        Returns
        -------
        dict[str, int]
            A dictionary of chromosome sizes.

        Raises
        ------
        GenomeError
            If chromosome sizes cannot be computed from the FASTA file.
        """
        if self._chrom_sizes is not None:
            return self._chrom_sizes

        if self._computed_chrom_sizes is None:
            self._computed_chrom_sizes = self._compute_chrom_sizes()
            logger.info(f"Computed chromosome sizes for {self.name}: {len(self._computed_chrom_sizes)} chromosomes")

        return self._computed_chrom_sizes

    def _compute_chrom_sizes(self) -> ChromSizes:
        """Compute chromosome sizes from FASTA file."""
        try:
            from pyfaidx import Fasta

            fasta = Fasta(str(self.fasta))
            chrom_sizes = {chr_name: len(fasta[chr_name]) for chr_name in fasta.keys()}
            fasta.close()

            return chrom_sizes
        except ImportError as e:
            raise GenomeError("pyfaidx is required to compute chromosome sizes from FASTA") from e
        except Exception as e:
            raise GenomeError(f"Failed to compute chromosome sizes from FASTA: {e}") from e

    def validate(self) -> bool:
        """
        Validate that all genome files exist and are accessible.

        Returns
        -------
        bool
            True if validation passes.

        Raises
        ------
        GenomeError
            If validation fails.
        """
        try:
            # Check that files exist
            _ = self.fasta
            _ = self.annotation
            _ = self.chrom_sizes

            logger.info(f"Genome {self.name} validation passed")
            return True
        except Exception as e:
            raise GenomeError(f"Genome {self.name} validation failed: {e}") from e

    def get_chromosome_list(self) -> list[str]:
        """
        Get a list of all chromosome names.

        Returns
        -------
        list[str]
            List of chromosome names.
        """
        return list(self.chrom_sizes.keys())

    def get_total_genome_size(self) -> int:
        """
        Get the total size of the genome in base pairs.

        Returns
        -------
        int
            Total genome size in base pairs.
        """
        return sum(self.chrom_sizes.values())

    def get_chromosome_size(self, chromosome: str) -> int:
        """
        Get the size of a specific chromosome.

        Parameters
        ----------
        chromosome : str
            Chromosome name.

        Returns
        -------
        int
            Chromosome size in base pairs.

        Raises
        ------
        KeyError
            If chromosome is not found.
        """
        if chromosome not in self.chrom_sizes:
            available = ', '.join(self.get_chromosome_list())
            raise KeyError(f"Chromosome {chromosome} not found. Available: {available}")
        return self.chrom_sizes[chromosome]

    def __repr__(self) -> str:
        """String representation of the genome."""
        return f"Genome(name='{self.name}', chromosomes={len(self.chrom_sizes)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        total_size = self.get_total_genome_size()
        return f"Genome '{self.name}' with {len(self.chrom_sizes)} chromosomes ({total_size:,} bp total)"

    @classmethod
    def get_available_genomes(cls) -> list[str]:
        """
        Get a list of all registered genome names.

        Returns
        -------
        list[str]
            List of available genome names.
        """
        return list(cls._registry.keys())

    @classmethod
    def get_genome(cls, name: str) -> 'Genome':
        """
        Get a registered genome by name.

        Parameters
        ----------
        name : str
            Genome name.

        Returns
        -------
        Genome
            The requested genome instance.

        Raises
        ------
        KeyError
            If genome is not found in registry.
        """
        if name not in cls._registry:
            available = ', '.join(cls.get_available_genomes())
            raise KeyError(f"Genome '{name}' not found. Available: {available}")
        return cls._registry[name]


# Human genome assemblies
_HG38_CHROM_SIZES = {
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
}

_MM39_CHROM_SIZES = {
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
}

_MM10_CHROM_SIZES = {
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
}


def _create_hg19_genome() -> Genome:
    """Create hg19 genome instance."""
    return Genome(
        name="hg19",
        fasta=lambda: register_datasets().fetch(
            "gencode_v41_GRCh37.fa.gz",
            processor=Decompress(method="gzip"),
            progressbar=True,
        ),
        annotation=lambda: register_datasets().fetch(
            "gencode_v41_GRCh37.gff3.gz",
            progressbar=True
        ),
    )


def _create_hg38_genome() -> Genome:
    """Create hg38 genome instance."""
    return Genome(
        name="hg38",
        fasta=lambda: register_datasets().fetch(
            "gencode_v41_GRCh38.fa.gz",
            processor=Decompress(method="gzip"),
            progressbar=True,
        ),
        annotation=lambda: register_datasets().fetch(
            "gencode_v41_GRCh38.gff3.gz",
            progressbar=True
        ),
        chrom_sizes=_HG38_CHROM_SIZES,
    )


def _create_mm39_genome() -> Genome:
    """Create mm39 genome instance."""
    return Genome(
        name="mm39",
        fasta=lambda: register_datasets().fetch(
            "gencode_vM30_GRCm39.fa.gz",
            processor=Decompress(method="gzip"),
            progressbar=True,
        ),
        annotation=lambda: register_datasets().fetch(
            "gencode_vM30_GRCm39.gff3.gz",
            progressbar=True
        ),
        chrom_sizes=_MM39_CHROM_SIZES,
    )


def _create_mm10_genome() -> Genome:
    """Create mm10 genome instance."""
    return Genome(
        name="mm10",
        fasta=lambda: register_datasets().fetch(
            "gencode_vM25_GRCm38.fa.gz",
            processor=Decompress(method="gzip"),
            progressbar=True,
        ),
        annotation=lambda: register_datasets().fetch(
            "gencode_vM25_GRCm38.gff3.gz",
            progressbar=True
        ),
        chrom_sizes=_MM10_CHROM_SIZES,
    )


# Create predefined genome instances
hg19 = _create_hg19_genome()
hg38 = _create_hg38_genome()
mm39 = _create_mm39_genome()
mm10 = _create_mm10_genome()

# Export commonly used genomes
__all__ = [
    "Genome",
    "GenomeError",
    "GenomeFileNotFoundError",
    "GenomeValidationError",
    "hg19",
    "hg38",
    "mm39",
    "mm10",
]
