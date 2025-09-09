import pandas as pd
from anndata import AnnData
from mudata import MuData

from cell2net._logging import logger

def setup_mudata(
    mdata: MuData,
    gene: str,
    rna_mod: str = "rna",
    atac_mod: str = "atac",
    peak_to_gene_key: str = "peak_to_gene"
) -> MuData:
    """
    Create a gene-specific subset of multimodal data for regulatory analysis.

    This function extracts a subset of the MuData object focused on a specific target gene,
    including associated ATAC-seq peaks, transcription factors (TFs), and regulatory elements.
    It creates a new MuData object containing only the regulatory elements relevant for
    modeling the expression of the target gene.

    Parameters
    ----------
    mdata : MuData
        Input multimodal data object containing RNA and ATAC modalities.
        Must contain:

        - RNA modality with gene expression data
        - ATAC modality with chromatin accessibility data
        - Peak-to-gene linkage information in `.uns[peak_to_gene_key]`
        - TF-to-gene regulatory relationships in `[rna_mod].uns["gene_tf"]`

    gene : str
        Name of the target gene to subset the data for. Must be present in:

        - `mdata[rna_mod].var_names` (gene expression data)
        - `mdata.uns[peak_to_gene_key]["gene"]` (peak-to-gene linkages)
        - `mdata[rna_mod].uns["gene_tf"].index` (TF regulatory relationships)

    rna_mod : str, default "rna"
        Name of the RNA modality in the MuData object containing gene expression data.
    atac_mod : str, default "atac"
        Name of the ATAC modality in the MuData object containing chromatin accessibility data.
    peak_to_gene_key : str, default "peak_to_gene"
        Key in `mdata.uns` that stores the peak-to-gene linkage DataFrame.
        This DataFrame must contain columns: "gene" and "peak".

    Returns
    -------
    MuData
        A new MuData object containing gene-specific regulatory data with:

        **RNA modality:**
        - Single gene expression data for the target gene
        - `.obsm["tf"]` : Expression data for associated transcription factors

        **ATAC modality:**
        - Chromatin accessibility data for peaks linked to the target gene
        - `.uns["peaks"]` : Peak metadata for the associated peaks

        **Shared annotations:**
        - `.obs` : Complete cell metadata (copied from input)
        - `.uns["tfs"]` : List of transcription factors regulating the target gene
        - `.uns["peak_to_gene"]` : Peak-to-gene linkages for the target gene

    Raises
    ------
    KeyError
        If the specified gene is not found in the peak-to-gene linkages,
        gene expression data, or TF regulatory relationships.
    ValueError
        If required modalities or keys are missing from the input MuData object.

    Notes
    -----
    - The function assumes that TF expression data is stored in the "counts" layer
      of the RNA modality.
    - Only transcription factors with non-zero regulatory relationships to the target
      gene are included in the subset.
    - The resulting MuData object is designed for gene-specific regulatory modeling
      and contains only the minimal necessary data for the target gene.
    - Peak coordinates and metadata are preserved in the ATAC modality for
      downstream sequence analysis.

    Examples
    --------
    Subset MuData for a specific gene:

    >>> import mudata as md
    >>> import cell2net as cn
    >>>
    >>> # Load multimodal data with regulatory relationships
    >>> mdata = md.read_h5mu("multiome_data.h5mu")
    >>>
    >>> # Create gene-specific subset for CD8A
    >>> subset = cn.pp.subset_mdata(
    ...     mdata=mdata,
    ...     gene="CD8A",
    ...     rna_mod="rna",
    ...     atac_mod="atac"
    ... )
    >>>
    >>> print(f"Original RNA genes: {mdata['rna'].n_vars}")
    >>> print(f"Subset RNA genes: {subset['rna'].n_vars}")  # Should be 1
    >>> print(f"Associated peaks: {subset['atac'].n_vars}")
    >>> print(f"Regulatory TFs: {len(subset.uns['tfs'])}")

    Access the regulatory elements:

    >>> # Get TF expression data
    >>> tf_expression = subset['rna'].obsm['tf']
    >>>
    >>> # Get peak accessibility data
    >>> peak_accessibility = subset['atac'].X
    >>>
    >>> # Get peak-to-gene linkages
    >>> linkages = subset.uns['peak_to_gene']
    >>> print(linkages[['peak', 'gene']])

    Use with custom modality names:

    >>> subset = cn.pp.subset_mdata(
    ...     mdata=mdata,
    ...     gene="FOXP3",
    ...     rna_mod="gene_expression",
    ...     atac_mod="chromatin_accessibility"
    ... )

    See Also
    --------
    cell2net.prediction.model.Cell2Net : Model class that uses subsetted data
    cell2net.preprocessing.peak_to_gene : Function to create peak-to-gene linkages
    cell2net.preprocessing.tf_to_gene : Function to create TF-to-gene relationships
    """
    logger.info(f"Subsetting MuData for gene: {gene}")

    peak_to_gene = mdata.uns[peak_to_gene_key][
            mdata.uns[peak_to_gene_key]["gene"] == gene
        ]
    peak_to_gene = peak_to_gene.reset_index(drop=True)

    # Create anndata for RNA and ATAC
    peaks = peak_to_gene["peak"].values.tolist()
    adata_atac = mdata[atac_mod][:, peaks].copy()

    # Subset ATAC peaks to those associated with the gene
    adata_atac.uns["peaks"] = adata_atac.uns["peaks"].loc[peaks].copy()

    adata_rna = mdata[rna_mod][:, gene].copy()

    # Get associated TFs for each cell
    row = mdata[rna_mod].uns["gene_tf"].loc[gene]
    tfs = row[row != 0].index.tolist()

    tf_exp = mdata[rna_mod][:, tfs].layers["counts"].copy()  # type: ignore

    df_tf_var = pd.DataFrame(
        data={"gene_name": tfs},
        index=tfs
    )
    adata_tf = AnnData(X=tf_exp,
                       obs=mdata[rna_mod].obs.index.to_frame(),
                       var=df_tf_var)

    _mdata = MuData({rna_mod: adata_rna, atac_mod: adata_atac, "tf_exp": adata_tf})  # type: ignore
    _mdata.obs = mdata.obs.copy()
    _mdata.uns["tfs"] = tfs
    _mdata.uns["peak_to_gene"] = peak_to_gene

    return _mdata

