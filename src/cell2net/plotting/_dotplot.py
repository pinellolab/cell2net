"""Functions to make dot plot"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scanpy.plotting import DotPlot

from ._utils import _VarNames, process_var_names

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize


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
    ax: Axes,
    cmap: Colormap | str | None,
    color_on: Literal["dot", "square"] = "dot",
    dot_max: float | None = None,
    dot_min: float | None = None,
    smallest_dot: float = 0.0,
    largest_dot: float = 200.0,
    size_exponent: float = 1.5,
    # edge_color: ColorLike | None = "black",
    # edge_lw: float | None,
    # grid: bool,
    # x_padding: float,
    # y_padding: float,
    # vmin: float | None,
    # vmax: float | None,
    # vcenter: float | None,
    # norm: Normalize | None,
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

    ax.scatter(x, y, **kwds)

    return ax


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
    make_dot_plot(dot_color=dot_color, dot_size=dot_size, ax=ax, cmap=cmap)

    if return_fig:
        return ax
    else:
        return None
