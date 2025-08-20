import os
import gzip

import pandas as pd
from mudata import MuData

from cell2net._logging import logger

def get_gene_tss_coord(gene_gtf: str, feature_type: str = "gene") -> pd.DataFrame:
    """
    Extract transcription start site (TSS) coordinates for genes from a GTF file.

    This function parses a GTF (Gene Transfer Format) file to extract the transcription
    start site (TSS) of genes or other specified features. It returns a pandas DataFrame
    with the chromosome, gene name, strand, and TSS for each gene.

    Parameters
    ----------
    gene_gtf :
        Path to the GTF file. The file can be plain text or gzip-compressed (".gz").
    feature_type :
        The type of feature to extract (e.g., "gene", "transcript").

    Returns
    -------
    A DataFrame containing the following columns:

        - `chrom`: Chromosome name (str).
        - `gene_name`: Gene name extracted from the "gene_name" attribute (str).
        - `strand`: Strand of the gene ('+' or '-') (str).
        - `tss`: Transcription start site position (int).

    Notes
    -----
        - This function assumes that the "gene_name" attribute is present in the GTF file's attributes field and is enclosed in double quotes.
        - The TSS is calculated as the start position for '+' strand genes and the end position for '-' strand genes.
        - Lines in the GTF file that do not have 9 columns or do not match the specified feature type are skipped.
        - Duplicate gene names are removed, keeping only the first occurrence.

    Examples
    --------
    >>> Extract TSS information for genes from a GTF file:
    >>> import pandas as pd
    >>> gene_gtf = "path/to/genes.gtf.gz"
    >>> df = get_gene_tss_coor(gene_gtf)
    >>> print(df.head())
        chrom  gene_name strand    tss
    0    chr1      GeneA      +   1000
    1    chr1      GeneB      -   2000
    2    chr2      GeneC      +   3000
    3    chr2      GeneD      -   4000
    """
    if gene_gtf.endswith(".gz"):
        file_handle = gzip.open(gene_gtf, "rt")
    else:
        file_handle = open(gene_gtf)

    gene_info = []
    for line in file_handle:
        if line.startswith("#"):
            continue  # Skip comment lines

        # Split the line into fields
        fields = line.strip().split("\t")
        if len(fields) != 9:
            continue  # Skip lines that don't have 9 columns

        # Extract relevant fields
        chrom = fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        attributes = fields[8]

        # We're only interested in gene features
        if fields[2] == feature_type:
            # Extract gene name from attributes (assuming it's under the 'gene_name' tag)
            gene_name = None
            for attr in attributes.split(";"):
                if "gene_name" in attr:
                    gene_name = attr.split('"')[1]
                    break

            # If no gene_name is found, continue to the next line
            if gene_name is None:
                continue

            # Determine TSS based on strand
            tss = start if strand == "+" else end

            # Append the gene information (chromosome, gene name, strand, TSS)
            gene_info.append([chrom, gene_name, strand, tss])

    # Convert gene_info into a DataFrame for easier manipulation
    df = pd.DataFrame(gene_info, columns=["chrom", "gene_name", "strand", "tss"])
    df = df.drop_duplicates(["gene_name"], keep="first")

    return df


def add_gene_tss_coord(
    mdata: MuData,
    gene_gtf: str,
    feature_type: str = "gene",
    mod_names: str = "rna",
    inplace: bool = True
) -> None | pd.DataFrame:
    """
    Add transcription start site (TSS) coordinates for genes to a MuData object.

    This function extracts TSS coordinates from a GTF file and adds them to the specified
    modality in a MuData object. It filters the TSS data to include only genes present
    in the modality's variable names (gene list). The function can either store the
    results in the MuData object or return them as a DataFrame.

    Parameters
    ----------
    mdata : MuData
        A MuData object containing multimodal single-cell data. The specified modality
        should contain gene expression data with gene names in `.var_names`.
    gene_gtf : str
        Path to the GTF (Gene Transfer Format) file containing gene annotations.
        The file can be plain text or gzip-compressed (".gz" extension).
        Must exist and be accessible.
    feature_type : str, default "gene"
        The type of genomic feature to extract from the GTF file.
        Common options include:

        - "gene" : Extract gene-level features
        - "transcript" : Extract transcript-level features
        - "exon" : Extract exon-level features
    mod_names : str, default "rna"
        Name of the modality in `mdata` that contains the gene expression data.
        This modality's `.var_names` will be used to filter the TSS coordinates.
        Must exist in `mdata.mod_names`.
    inplace : bool, default True
        Whether to store the TSS coordinates in the MuData object.

        - If True: Stores results in `mdata[mod_names].uns["gene_tss_coord"]` and returns None
        - If False: Returns the DataFrame without modifying the MuData object

    Returns
    -------
    None or pd.DataFrame
        - If `inplace=True`: Returns None. The TSS coordinates are stored in
          `mdata[mod_names].uns["gene_tss_coord"]`.
        - If `inplace=False`: Returns a DataFrame with TSS coordinates.

        The DataFrame (whether stored or returned) contains:

        - `chrom` : str - Chromosome name
        - `gene_name` : str - Gene name (filtered to match modality gene names)
        - `strand` : str - Strand orientation ('+' or '-')
        - `tss` : int - Transcription start site position (1-based genomic coordinate)

    Raises
    ------
    FileNotFoundError
        If the specified GTF file does not exist (logged as error, returns None).
    KeyError
        If the specified modality is not found in `mdata.mod_names` (logged as error, returns None).

    Notes
    -----
    - The function uses `get_gene_tss_coord()` internally to parse the GTF file.
    - TSS positions are calculated based on strand orientation:

      * For '+' strand genes: TSS = start position
      * For '-' strand genes: TSS = end position

    - Only genes present in the modality's `.var_names` are included in the final result.
    - The function assumes that gene names in the GTF file match those in the modality.
    - If duplicate gene names exist in the GTF, only the first occurrence is kept.
    - The TSS coordinates are stored with 1-based genomic coordinates.

    Examples
    --------
    Add TSS coordinates to RNA modality and store in MuData:

    >>> import mudata as md
    >>> import cell2net as cn
    >>>
    >>> # Load multimodal data
    >>> mdata = md.read_h5mu("multiome_data.h5mu")
    >>>
    >>> # Add TSS coordinates from GTF file
    >>> cn.pp.add_gene_tss_coord(
    ...     mdata=mdata,
    ...     gene_gtf="/path/to/genes.gtf.gz",
    ...     mod_names="rna"
    ... )
    >>>
    >>> # Access the stored TSS coordinates
    >>> tss_df = mdata["rna"].uns["gene_tss_coord"]
    >>> print(f"TSS coordinates for {len(tss_df)} genes")

    Return TSS coordinates without modifying MuData:

    >>> tss_df = cn.pp.add_gene_tss_coord(
    ...     mdata=mdata,
    ...     gene_gtf="/path/to/annotation.gtf",
    ...     feature_type="gene",
    ...     mod_names="rna",
    ...     inplace=False
    ... )
    >>> print(tss_df.head())

    Use with transcript-level features:

    >>> cn.pp.add_gene_tss_coord(
    ...     mdata=mdata,
    ...     gene_gtf="/path/to/transcripts.gtf.gz",
    ...     feature_type="transcript",
    ...     mod_names="rna"
    ... )

    See Also
    --------
    get_gene_tss_coord : Extract TSS coordinates directly from GTF file
    """
    # check if gene_gtf exists
    if not os.path.exists(gene_gtf):
        logger.error(f"Gene GTF file {gene_gtf} does not exist.")
        return None

    # check if mod_names is in mdata.mod_names
    if mod_names not in mdata.mod_names:
        logger.error(f"Modality {mod_names} not found in mdata.mod_names.")
        return None

    adata = mdata[mod_names]

    logger.info(f"Adding gene TSS coordinates from {gene_gtf} to {mod_names} modality.")

    df = get_gene_tss_coord(gene_gtf=gene_gtf, feature_type=feature_type)

    df = df[df["gene_name"].isin(adata.var_names)].reset_index(drop=True)

    if inplace:
        adata.uns["gene_tss_coord"] = df
        return None
    else:
        return df
