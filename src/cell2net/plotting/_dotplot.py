from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

_VarNames = str | Sequence[str]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap


def _prepare_var_names():
    pass


def prepare_dataframes(df: pd.DataFrame):
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


def _dot_plot(
    dot_size: pd.DataFrame,
    dot_color: pd.DataFrame,
    dot_ax: Axes,
    cmap: Colormap | str | None = "Reds",
    color_on: Literal["dot", "square"] = "dot",
    dot_max: float | None = None,
    dot_min: float | None = None,
    standard_scale: Literal["var", "group"] | None = "var",
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
    The indices and columns of the data frame are used to label the output image

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

    if standard_scale == "group":
        dot_color = dot_color.sub(dot_color.min(1), axis=0)
        dot_color = dot_color.div(dot_color.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        dot_color -= dot_color.min(0)
        dot_color = (dot_color / dot_color.max(0)).fillna(0)
    elif standard_scale is None:
        pass

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

    return dot_ax


def tf_dotplot(
    df: pd.DataFrame,
    ax: Axes,
    cmap: Colormap | str | None,
    var_names: _VarNames | Mapping[str, _VarNames],
    categories_order: Sequence[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    standard_scale="var",
    dendrogram=True,
) -> None:
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
    #
    width, height = figsize if figsize is not None else (None, None)

    dot_color, dot_size = prepare_dataframes(df)

    # subset the dataframe
    has_var_groups = False
    if isinstance(var_names, Mapping):
        var_group_labels = []
        _var_names = []
        var_group_positions = []
        start = 0
        for label, vars_list in var_names.items():
            if isinstance(vars_list, str):
                vars_list = [vars_list]
            # use list() in case var_list is a numpy array or pandas series
            _var_names.extend(list(vars_list))
            var_group_labels.append(label)
            var_group_positions.append((start, start + len(vars_list) - 1))
            start += len(vars_list)

        var_names = _var_names
        var_group_labels = var_group_labels
        var_group_positions = var_group_positions
        has_var_groups = True

    elif isinstance(var_names, str):
        var_names = [var_names]

    dot_color = dot_color[var_names]
    dot_size = dot_size[var_names]

    if standard_scale == "group":
        dot_color = dot_color.sub(dot_color.min(1), axis=0)
        dot_color = dot_color.div(dot_color.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        dot_color -= dot_color.min(0)
        dot_color = (dot_color / dot_color.max(0)).fillna(0)
    elif standard_scale is None:
        pass

    # make scatter plot in which
    # x = var_names
    # y = groupby category
    # size = number of target genes
    # color = mean regulation activity

    # +0.5 in y and x to set the dot center at 0.5 multiples
    # this facilitates dendrogram and totals alignment for
    # matrixplot, dotplot and stackec_violin using the same coordinates.

    pass
