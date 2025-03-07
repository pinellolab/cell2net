import gzip

import pandas as pd
from mudata import MuData


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
) -> None:
    """
    Add the TSS coordinates of genes to mdata[mod_names].uns.

    Parameters
    ----------
    mdata :
        Input MuData object containing gene expression
    gene_gtf :
        GTF file including gene annotation, which should have 9 columns.
    feature_type :
        Which feature type in the GTF file to use. Default: gene
    """
    assert mod_names in mdata.mod_names, f"Cannot find modality: {mod_names}"
    adata = mdata[mod_names]

    df = get_gene_tss_coord(gene_gtf=gene_gtf, feature_type=feature_type)

    adata.uns["gene_tss_coord"] = df[df["gene_name"].isin(adata.var_names)].reset_index(
        drop=True
    )

    return None
