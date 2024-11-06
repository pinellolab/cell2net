import gzip

import pandas as pd
from mudata import MuData


def get_gene_tss_coor(gene_gtf: str, feature_type: str = "gene") -> pd.DataFrame:
    """
    Extract

    Parameters
    ----------
    gene_gtf : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
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
    Add the TSS coordinates of genes to mdata.

    Parameters
    ----------
    mdata : MuData
        Input MuData object containing gene expression
    gene_gtf : str
        GTF file including gene annotation, which should have 9 columns.
    feature_type : str
    """
    assert mod_names in mdata.mod_names, f"Cannot find modality: {mod_names}"
    adata = mdata[mod_names]

    df = get_gene_tss_coor(gene_gtf=gene_gtf, feature_type=feature_type)

    adata.uns["gene_tss_coord"] = df[df["gene_name"].isin(adata.var_names)]

    return None
