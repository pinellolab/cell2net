"""Functions to make dot plot"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scanpy.plotting._baseplot_class import BasePlot
from scanpy.plotting._utils import ColorLike, _AxesSubplot, check_colornorm, fix_kwds

from ._utils import _VarNames, process_var_names

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize


class DotPlot(BasePlot):
    DEFAULT_SAVE_PREFIX = "dotplot_"
    # default style parameters
    DEFAULT_COLORMAP = "Reds"
    DEFAULT_COLOR_ON = "dot"
    DEFAULT_DOT_MAX = None
    DEFAULT_DOT_MIN = None
    DEFAULT_SMALLEST_DOT = 0.0
    DEFAULT_LARGEST_DOT = 200.0
    DEFAULT_DOT_EDGECOLOR = "black"
    DEFAULT_DOT_EDGELW = 0.2
    DEFAULT_SIZE_EXPONENT = 1.5

    # default legend parameters
    DEFAULT_SIZE_LEGEND_TITLE = "Fraction of cells\nin group (%)"
    DEFAULT_COLOR_LEGEND_TITLE = "Mean expression\nin group"
    DEFAULT_LEGENDS_WIDTH = 1.5  # inches
    DEFAULT_PLOT_X_PADDING = 0.8  # a unit is the distance between two x-axis ticks
    DEFAULT_PLOT_Y_PADDING = 1.0  # a unit is the distance between two y-axis ticks

    def __init__(
        self,
        dot_color_df: pd.DataFrame,
        dot_size_df: pd.DataFrame,
        categories_order: Sequence[str] | None = None,
        ax: _AxesSubplot | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        vcenter: float | None = None,
        norm: Normalize | None = None,
        **kwds,
    ):

        # Because genes (columns) can be duplicated (e.g. when the
        # same gene is reported as marker gene in two clusters)
        # they need to be removed first,
        # otherwise, the duplicated genes are further duplicated when reordering
        # Eg. A df with columns ['a', 'b', 'a'] after reordering columns
        # with df[['a', 'a', 'b']], results in a df with columns:
        # ['a', 'a', 'a', 'a', 'b']

        unique_var_names, unique_idx = np.unique(
            dot_color_df.columns, return_index=True
        )

        # remove duplicate columns
        if len(unique_var_names) != len(self.var_names):
            dot_color_df = dot_color_df.iloc[:, unique_idx]

        # get the same order for rows and columns in the dot_color_df
        # using the order from the doc_size_df
        dot_color_df = dot_color_df.loc[dot_size_df.index][dot_size_df.columns]

        self.categories = dot_color_df.index.tolist()

        self.dot_color_df, self.dot_size_df = (
            df.loc[
                categories_order if categories_order is not None else self.categories
            ]  # type: ignore
            for df in (dot_color_df, dot_size_df)
        )


def prepare_dataframes_for_tf_dotplot(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the color and size dataframe for dot plot

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    _type_
        _description_
    """
    # get average regulation of each tf within cell types
    # used for dot color
    dot_color_df = df.groupby(["tf", "cell_type_v2"])["avg_attr"].sum().reset_index()
    dot_color_df = dot_color_df.pivot_table(
        index="cell_type_v2", columns="tf", values="avg_attr"
    )
    dot_color_df = dot_color_df.fillna(0)
    dot_color_df = dot_color_df.rename_axis(None, axis=0)
    dot_color_df = dot_color_df.rename_axis(None, axis=1)

    # get number of target genes of each within cell types
    # used for dot size
    dot_size_df = (
        df.groupby(["tf", "cell_type_v2"])["avg_attr"].count().reset_index(name="count")
    )
    dot_size_df = dot_size_df.pivot_table(
        index="cell_type_v2", columns="tf", values="count"
    )
    dot_size_df = dot_size_df.fillna(0)
    dot_size_df = dot_size_df.rename_axis(None, axis=0)
    dot_size_df = dot_size_df.rename_axis(None, axis=1)

    return dot_color_df, dot_size_df


def make_dot_plot(
    dot_size: pd.DataFrame,
    dot_color: pd.DataFrame,
    dot_ax: Axes,
    cmap: Colormap | str | None,
    color_on: Literal["dot", "square"],
    dot_max: float | None,
    dot_min: float | None,
    smallest_dot: float,
    largest_dot: float,
    size_exponent: float,
    edge_color: ColorLike | None,
    edge_lw: float | None,
    grid: bool,
    x_padding: float,
    y_padding: float,
    vmin: float | None,
    vmax: float | None,
    vcenter: float | None,
    norm: Normalize | None,
    **kwds,
):
    """
    Makes a *dot plot* given two data frames.

    One contains the doc size and other containing the dot color.
    They should have the same indices and columns, which are used
    to label the output image.

    This function is modified from scanpy.plotting.Dotplot

    Parameters
    ----------
    dot_size : pd.DataFrame
        Data frame containing the dot_size.
    dot_color : pd.DataFrame
        Data frame containing the dot_color, should have the same,
            shape, columns and indices as dot_size.
    dot_ax : Axes
        matplotlib axis
    """
    assert dot_size.shape == dot_color.shape, (
        "please check that dot_size " "and dot_color dataframes have the same shape"
    )

    assert list(dot_size.index) == list(dot_color.index), (
        "please check that dot_size " "and dot_color dataframes have the same index"
    )

    assert list(dot_size.columns) == list(dot_color.columns), (
        "please check that the dot_size "
        "and dot_color dataframes have the same columns"
    )

    y, x = np.indices(dot_color.shape)
    y = y.flatten() + 0.5
    x = x.flatten() + 0.5
    frac = dot_size.values.flatten()
    mean_flat = dot_color.values.flatten()
    cmap = plt.get_cmap(name=cmap)  # type: ignore

    if dot_max is None:
        dot_max = np.ceil(max(frac) * 10) / 10
    else:
        if dot_max < 0 or dot_max > 1:
            raise ValueError("`dot_max` value has to be between 0 and 1")
    if dot_min is None:
        dot_min = 0
    else:
        if dot_min < 0 or dot_min > 1:
            raise ValueError("`dot_min` value has to be between 0 and 1")

    if dot_min != 0 or dot_max != 1:
        # clip frac between dot_min and  dot_max
        frac = np.clip(frac, dot_min, dot_max)
        old_range = dot_max - dot_min  # type: ignore
        # re-scale frac between 0 and 1
        frac = (frac - dot_min) / old_range

    size = frac**size_exponent
    # rescale size to match smallest_dot and largest_dot
    size = size * (largest_dot - smallest_dot) + smallest_dot
    normalize = check_colornorm(vmin, vmax, vcenter, norm)

    color = cmap(normalize(mean_flat))  # type: ignore

    kwds = fix_kwds(
        kwds,
        s=size,
        color=color,
        linewidth=edge_lw,
        edgecolor=edge_color,
    )

    dot_ax.scatter(x, y, **kwds)
    y_ticks = np.arange(dot_color.shape[0]) + 0.5
    dot_ax.set_yticks(y_ticks)
    dot_ax.set_yticklabels(
        [dot_color.index[idx] for idx, _ in enumerate(y_ticks)], minor=False
    )
    x_ticks = np.arange(dot_color.shape[1]) + 0.5
    dot_ax.set_xticks(x_ticks)
    dot_ax.set_xticklabels(
        [dot_color.columns[idx] for idx, _ in enumerate(x_ticks)],
        rotation=90,
        ha="center",
        minor=False,
    )
    dot_ax.tick_params(axis="both", labelsize="small")
    dot_ax.grid(visible=False)

    # to be consistent with the heatmap plot, is better to
    # invert the order of the y-axis, such that the first group is on
    # top
    dot_ax.set_ylim(dot_color.shape[0], 0)
    dot_ax.set_xlim(0, dot_color.shape[1])

    if color_on == "dot":
        # add padding to the x and y lims when the color is not in the square
        # default y range goes from 0.5 to num cols + 0.5
        # and default x range goes from 0.5 to num rows + 0.5, thus
        # the padding needs to be corrected.
        x_padding = x_padding - 0.5
        y_padding = y_padding - 0.5
        dot_ax.set_ylim(dot_color.shape[0] + y_padding, -y_padding)
        dot_ax.set_xlim(-x_padding, dot_color.shape[1] + x_padding)

    if grid:
        dot_ax.grid(visible=True, color="gray", linewidth=0.1)
        dot_ax.set_axisbelow(True)

    return dot_ax


def tf_dotplot(
    df: pd.DataFrame,
    var_names: _VarNames | Mapping[str, _VarNames],
    categories_order: Sequence[str] | None = None,
    standard_scale: Literal["var", "group"] | None = None,
    title: str | None = None,
    colorbar_title: str | None = DotPlot.DEFAULT_COLOR_LEGEND_TITLE,
    size_title: str | None = DotPlot.DEFAULT_SIZE_LEGEND_TITLE,
    figsize: tuple[float, float] | None = None,
    dendrogram: bool | str = False,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    var_group_rotation: float | None = None,
    layer: str | None = None,
    swap_axes: bool | None = False,
    dot_color_df: pd.DataFrame | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    ax: Axes | None = None,
    return_fig: bool | None = False,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    # Style parameters
    cmap: Colormap | str | None = DotPlot.DEFAULT_COLORMAP,
    dot_max: float | None = DotPlot.DEFAULT_DOT_MAX,
    dot_min: float | None = DotPlot.DEFAULT_DOT_MIN,
    smallest_dot: float = DotPlot.DEFAULT_SMALLEST_DOT,
    **kwds,
) -> None | Axes:
    """
    Makes a *dot plot* of the regulation activities of `var_names`.

    This function provides a convenient interface to the :class:`~scanpy.pl.DotPlot`
    class. If you need more flexibility, you should use :class:`~scanpy.pl.DotPlot`
    directly.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    var_names : _type_
        _description_
    standard_scale : str, optional
        _description_, by default "var"
    dendrogram : bool, optional
        _description_, by default True
    """
    size_legend_title = "Number of target genes"
    color_legend_title = "Mean regulation activity"

    # prepare data for plotting
    dot_color, dot_size = prepare_dataframes_for_tf_dotplot(df)

    # process variable names and subset dataframe
    var_names, var_group_labels, var_group_positions, has_var_groups = (
        process_var_names(var_names)
    )
    dot_color = dot_color[var_names]
    dot_size = dot_size[var_names]

    # normalize the dataframe for dot color
    if standard_scale == "group":
        dot_color = dot_color.sub(dot_color.min(1), axis=0)
        dot_color = dot_color.div(dot_color.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        dot_color -= dot_color.min(0)
        dot_color = (dot_color / dot_color.max(0)).fillna(0)
    elif standard_scale is None:
        pass

    # get dot
    # set figure size
    width, height = figsize if figsize is not None else (None, None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # make dot plot
    # make_dot_plot(dot_color=dot_color, dot_size=dot_size, ax=ax, cmap=cmap)

    # if return_fig:
    #     return ax
    # else:
    #     return None
