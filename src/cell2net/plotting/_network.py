import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from packaging.version import Version
from scanpy.plotting._utils import savefig_or_show

from cell2net._logging import logger


def tf_gene_network(df: pd.DataFrame,
                    tfs: str | list[str],
                    n_target: int = 10,
                    node_size: float = 10,
                    edge_width: float = 1, figsize=(10, 10),
                    show: bool | None = None,
                    save: str | bool | None = None,
                    save_prefix: str = "tf_gene_net_",
                    return_fig: bool = True):

    # check if igraph is correctly installed
    try:
        import igraph as ig
    except ImportError:
        logger.error("igraph is not installed. Please install it with: pip install igraph")

    if Version(ig.__version__) < Version("0.10.0"):
        logger.error("igraph version needs to be at least 0.10.0. Please install it with: pip install igraph==0.10.0")

    # subset the dataframe to only include the TFs and target genes
    # df = df.loc[df.index.isin(tfs + n_target)]

    # create edge list from long-formated dataframe
    # edges = df.reset_index().values
    # edges = [(x[0], x[1], x[2]) for x in edges]

    # create the graph
    g = ig.Graph(n=10, edges=[[0, 1], [2, 3], [3, 4]], directed=True)

    fig = plt.figure(figsize=(3, 3), dpi=150, tight_layout=True)
    gs = mpl.gridspec.GridSpec(5, 4, height_ratios=[1, 1, 1, 1, 0.10]) # type: ignore
    ax1 = fig.add_subplot(gs[:-1, :])
    ax2 = fig.add_subplot(gs[-1, 0])
    ax3 = fig.add_subplot(gs[-1, -1])
    ig.plot(g, target=ax1, layout='kk')

    if return_fig:
        return fig
    else:
        savefig_or_show(save_prefix, show=show, save=save)
        # show = settings.autoshow if show is None else show
        # if not show:
        #     return dp.get_axes()


