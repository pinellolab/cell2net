from ._dotplot import (
    DotPlot,
    prepare_dataframes_for_dotplot,
    tf_dotplot,
)
from ._heatmap import peak_to_gene_heatmap
from ._lineplot import train_history
from ._motif import motif_logo
from ._scatterplot import n_peaks_per_gene, tf_activity_variance
from ._utils import check_if_igraph, process_var_names
