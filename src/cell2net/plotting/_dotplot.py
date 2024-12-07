"""Functions to make dot plot"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scanpy._utils import Empty, _empty
from scanpy.plotting._baseplot_class import BasePlot
from scanpy.plotting._utils import ColorLike, _AxesSubplot, check_colornorm, fix_kwds

from cell2net._logging import logger

from ._utils import _VarNames, process_var_names

if TYPE_CHECKING:
    from typing import Literal

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

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
DEFAULT_SIZE_LEGEND_TITLE = "Number of target \ngenes in group"
DEFAULT_COLOR_LEGEND_TITLE = "Mean regulation activity\nin group"
DEFAULT_LEGENDS_WIDTH = 1.5  # inches
DEFAULT_PLOT_X_PADDING = 0.8  # a unit is the distance between two x-axis ticks
DEFAULT_PLOT_Y_PADDING = 1.0  # a unit is the distance between two y-axis ticks

# gridspec parameter. Sets the space between mainplot, dendrogram and legend
DEFAULT_WSPACE = 0


class DotPlot(BasePlot):
    """
    Class for *dot plot*

    This class is modified from scanpy.plotting.Dotplot by taking two pandas.DataFrame
    as inputs instead of an anndata object.

    One dataframe contains the dot color and another one contains the dot size.
    They should have the same indices and columns, which are used to label the output image.

    Parameters
    ----------
    dot_color_df : pd.DataFrame
        Input dataframe, used for dot color
    dot_size_df: pd.DataFrame
        Input dataframe, used for dot size
    var_names: _VarNames | Mapping[str, _VarNames]
        Variable names for plotting, can be a string, a list of strings, or a dictionary where
    """

    def __init__(
        self,
        dot_color_df: pd.DataFrame,
        dot_size_df: pd.DataFrame,
        var_names: _VarNames | Mapping[str, _VarNames],
        standard_scale: Literal["var", "group"] | None = None,
        var_group_positions: Sequence[tuple[int, int]] | None = None,
        var_group_labels: Sequence[str] | None = None,
        var_group_rotation: float | None = None,
        categories_order: Sequence[str] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        ax: _AxesSubplot | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        vcenter: float | None = None,
        norm: Normalize | None = None,
        **kwds,
    ):
        self.var_names = var_names
        self.var_group_labels = var_group_labels
        self.var_group_positions = var_group_positions
        self.var_group_rotation = var_group_rotation
        self.width, self.height = figsize if figsize is not None else (None, None)

        self.has_var_groups = (
            True
            if var_group_positions is not None and len(var_group_positions) > 0
            else False
        )

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

        self.standard_scale = standard_scale

        if categories_order is not None:
            if set(self.categories) != set(categories_order):
                logger.error(
                    "Please check that the categories given by "
                    "the `order` parameter match the categories that "
                    "want to be reordered.\n\n"
                    "Mismatch: "
                    f"{set(self.categories).difference(categories_order)}\n\n"
                    f"Given order categories: {categories_order}\n\n"
                    f"Find categories: {self.categories}\n"
                )
                return

        # Set fig title
        self.fig_title = title

        # set default values for legend
        self.color_legend_title = DEFAULT_COLOR_LEGEND_TITLE
        self.legends_width = DEFAULT_LEGENDS_WIDTH

        # Set default style parameters
        self.cmap = DEFAULT_COLORMAP
        self.dot_max = DEFAULT_DOT_MAX
        self.dot_min = DEFAULT_DOT_MIN
        self.smallest_dot = DEFAULT_SMALLEST_DOT
        self.largest_dot = DEFAULT_LARGEST_DOT
        self.color_on = DEFAULT_COLOR_ON
        self.size_exponent = DEFAULT_SIZE_EXPONENT
        self.grid = False
        self.plot_x_padding = DEFAULT_PLOT_X_PADDING
        self.plot_y_padding = DEFAULT_PLOT_Y_PADDING

        # style default parameters
        self.are_axes_swapped = False
        self.categories_order = categories_order
        self.var_names_idx_order = None

        self.wspace = DEFAULT_WSPACE

        self.group_extra_size = 0
        self.plot_group_extra = None

        # after .render() is called the fig value is assigned and ax_dict
        # contains a dictionary of the axes used in the plot
        self.fig = None
        self.ax_dict = None
        self.ax = ax

    def style(
        self,
        *,
        cmap: Colormap | str | None | Empty = _empty,
        color_on: Literal["dot", "square"] | Empty = _empty,
        dot_max: float | None | Empty = _empty,
        dot_min: float | None | Empty = _empty,
        smallest_dot: float | Empty = _empty,
        largest_dot: float | Empty = _empty,
        dot_edge_color: ColorLike | None | Empty = _empty,
        dot_edge_lw: float | None | Empty = _empty,
        size_exponent: float | Empty = _empty,
        grid: bool | Empty = _empty,
        x_padding: float | Empty = _empty,
        y_padding: float | Empty = _empty,
    ):
        """
        Modifies plot visual parameters

        Parameters
        ----------
        cmap : Colormap | str | None | Empty, optional
            String denoting matplotlib color map, by default _empty
        color_on : Literal["dot", "square"] | Empty, optional
            By default the color map is applied to the color of the ``"dot"``.
            Optionally, the colormap can be applied to a ``"square"`` behind the dot,
            in which case the dot is transparent and only the edge is shown, by default _empty
        dot_max : float | None | Empty, optional
            If ``None``, the maximum dot size is set to the maximum fraction value found (e.g. 0.6).
            If given, the value should be a number between 0 and 1.
            All fractions larger than dot_max are clipped to this value, by default _empty
        dot_min : float | None | Empty, optional
            If ``None``, the minimum dot size is set to 0.
            If given, the value should be a number between 0 and 1.
            All fractions smaller than dot_min are clipped to this value, by default _empty
        smallest_dot : float | Empty, optional
            All expression fractions with `dot_min` are plotted with this size, by default _empty
        largest_dot : float | Empty, optional
            All expression fractions with `dot_max` are plotted with this size, by default _empty
        dot_edge_color : ColorLike | None | Empty, optional
            Dot edge color.
            When `color_on='dot'`, ``None`` means no edge.
            When `color_on='square'`, ``None`` means that
            the edge color is white for darker colors and black for lighter background square, by default _empty
        dot_edge_lw : float | None | Empty, optional
            Dot edge line width.
            When `color_on='dot'`, ``None`` means no edge.
            When `color_on='square'`, ``None`` means a line width of 1.5, by default _empty
        size_exponent : float | Empty, optional
            Dot size is computed as:
            fraction  ** size exponent and afterwards scaled to match the
            `smallest_dot` and `largest_dot` size parameters.
            Using a different size exponent changes the relative sizes of the dots
            to each other, by default _empty
        grid : bool | Empty, optional
            Set to true to show grid lines. By default grid lines are not shown.
            Further configuration of the grid lines can be achieved directly on the
            returned ax, by default _empty
        x_padding : float | Empty, optional
            Space between the plot left/right borders and the dots center. A unit
            is the distance between the x ticks. Only applied when color_on = dot, by default _empty
        y_padding : float | Empty, optional
            Space between the plot top/bottom borders and the dots center. A unit is
            the distance between the y ticks. Only applied when color_on = dot, by default _empty
        """
        super().style(cmap=cmap)  # type: ignore

        if dot_max is not _empty:
            self.dot_max = dot_max
        if dot_min is not _empty:
            self.dot_min = dot_min
        if smallest_dot is not _empty:
            self.smallest_dot = smallest_dot
        if largest_dot is not _empty:
            self.largest_dot = largest_dot
        if color_on is not _empty:
            self.color_on = color_on
        if size_exponent is not _empty:
            self.size_exponent = size_exponent
        if dot_edge_color is not _empty:
            self.dot_edge_color = dot_edge_color
        if dot_edge_lw is not _empty:
            self.dot_edge_lw = dot_edge_lw
        if grid is not _empty:
            self.grid = grid
        if x_padding is not _empty:
            self.plot_x_padding = x_padding
        if y_padding is not _empty:
            self.plot_y_padding = y_padding

        pass

    def legend(
        self,
        *,
        show: bool | None = True,
        show_size_legend: bool | None = True,
        show_colorbar: bool | None = True,
        size_title: str | None = DEFAULT_SIZE_LEGEND_TITLE,
        colorbar_title: str | None = DEFAULT_COLOR_LEGEND_TITLE,
        width: float | None = DEFAULT_LEGENDS_WIDTH,
    ):
        r"""
        Configures dot size and the colorbar legends

        Parameters
        ----------
        show : bool | None, optional
            Set to `False` to hide the default plot of the legends.
            This sets the legend width to zero, which will result in a wider main plot., by default True
        show_size_legend : bool | None, optional
            Set to `False` to hide the dot size legend, by default True
        show_colorbar : bool | None, optional
            Set to `False` to hide the colorbar legend, by default True
        size_title : str | None, optional
            Title for the dot size legend. Use '\\n' to add line breaks.
            Appears on top of dot sizes, by default DEFAULT_SIZE_LEGEND_TITLE
        colorbar_title : str | None, optional
            Title for the color bar. Use '\\n' to add line breaks.
            Appears on top of the color bar, by default DEFAULT_COLOR_LEGEND_TITLE
        width : float | None, optional
            Width of the legends area. The unit is the same as in matplotlib (inches), by default DEFAULT_LEGENDS_WIDTH
        """
        if not show:
            # turn of legends by setting width to 0
            self.legends_width = 0
        else:
            self.color_legend_title = colorbar_title
            self.size_title = size_title
            self.legends_width = width
            self.show_size_legend = show_size_legend
            self.show_colorbar = show_colorbar

        return self

    def _mainplot(self, ax: Axes):
        # work on a copy of the dataframes. This is to avoid changes
        # on the original data frames after repetitive calls to the
        # DotPlot object, for example once with swap_axes and other without

        _color_df = self.dot_color_df.copy()
        _size_df = self.dot_size_df.copy()
        if self.var_names_idx_order is not None:
            _color_df = _color_df.iloc[:, self.var_names_idx_order]
            _size_df = _size_df.iloc[:, self.var_names_idx_order]

        if self.categories_order is not None:
            _color_df = _color_df.loc[self.categories_order, :]
            _size_df = _size_df.loc[self.categories_order, :]

        if self.are_axes_swapped:
            _size_df = _size_df.T
            _color_df = _color_df.T
        self.cmap = self.kwds.pop("cmap", self.cmap)

        normalize, dot_min, dot_max = self._dotplot(
            _size_df,
            _color_df,
            ax,
            cmap=self.cmap,
            color_on=self.color_on,  # type: ignore
            dot_max=self.dot_max,
            dot_min=self.dot_min,
            standard_scale=self.standard_scale,  # type: ignore
            edge_color=self.dot_edge_color,
            edge_lw=self.dot_edge_lw,
            smallest_dot=self.smallest_dot,
            largest_dot=self.largest_dot,
            size_exponent=self.size_exponent,
            grid=self.grid,
            x_padding=self.plot_x_padding,
            y_padding=self.plot_y_padding,
            vmin=self.vboundnorm.vmin,
            vmax=self.vboundnorm.vmax,
            vcenter=self.vboundnorm.vcenter,
            norm=self.vboundnorm.norm,
            **self.kwds,
        )

        self.dot_min, self.dot_max = dot_min, dot_max
        return normalize

    @staticmethod
    def _dotplot(
        dot_size: pd.DataFrame,
        dot_color: pd.DataFrame,
        dot_ax: Axes,
        *,
        cmap: Colormap | str | None,
        color_on: Literal["dot", "square"],
        dot_max: float | None,
        dot_min: float | None,
        standard_scale: Literal["var", "group"] | None,
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

        # make scatter plot in which
        # x = var_names
        # y = groupby category
        # size = fraction
        # color = mean expression

        # +0.5 in y and x to set the dot center at 0.5 multiples
        # this facilitates dendrogram and totals alignment for
        # matrixplot, dotplot and stackec_violin using the same coordinates.
        y, x = np.indices(dot_color.shape)
        y = y.flatten() + 0.5
        x = x.flatten() + 0.5
        frac = dot_size.values.flatten()
        mean_flat = dot_color.values.flatten()
        cmap = plt.get_cmap(cmap)  # type: ignore
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

        if color_on == "square":
            if edge_color is None:
                from seaborn.utils import relative_luminance

                # use either black or white for the edge color
                # depending on the luminance of the background
                # square color
                edge_color = []  # type: ignore
                for color_value in cmap(normalize(mean_flat)):  # type: ignore
                    lum = relative_luminance(color_value)
                    edge_color.append(".15" if lum > 0.408 else "w")  # type: ignore

            edge_lw = 1.5 if edge_lw is None else edge_lw

            # first make a heatmap similar to `sc.pl.matrixplot`
            # (squares with the asigned colormap). Circles will be plotted
            # on top
            dot_ax.pcolor(dot_color.values, cmap=cmap, norm=normalize)  # type: ignore
            for axis in ["top", "bottom", "left", "right"]:
                dot_ax.spines[axis].set_linewidth(1.5)
            kwds = fix_kwds(
                kwds,
                s=size,
                linewidth=edge_lw,
                facecolor="none",
                edgecolor=edge_color,
            )
            dot_ax.scatter(x, y, **kwds)
        else:
            edge_color = "none" if edge_color is None else edge_color
            edge_lw = 0.0 if edge_lw is None else edge_lw

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

        return normalize, dot_min, dot_max


def prepare_dataframes_for_dotplot(
    df: pd.DataFrame,
    tf_col: str = "tf",
    group_col: str = "cell_type",
    activity_col: str = "activity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the color and size dataframe for dot plot

    Parameters
    ----------
    df : pd.DataFrame
        Input data, should be a DataFrame with at least three columns with following format
    tf_col : str
        Column name for TFs
    group_col : str
        Column name for groups
    activity_col : str
        Column name for regulation activity

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        _description_
    """
    # get average regulation of each tf within cell types
    # used for dot color
    dot_color_df = df.groupby([tf_col, group_col])[activity_col].sum().reset_index()
    dot_color_df = dot_color_df.pivot_table(
        index=group_col, columns=tf_col, values=activity_col
    )
    dot_color_df = dot_color_df.fillna(0)
    dot_color_df = dot_color_df.rename_axis(None, axis=0)
    dot_color_df = dot_color_df.rename_axis(None, axis=1)

    # get number of target genes of each within cell types
    # used for dot size
    dot_size_df = (
        df.groupby([tf_col, group_col])[activity_col].count().reset_index(name="count")
    )
    dot_size_df = dot_size_df.pivot_table(
        index=group_col, columns=tf_col, values="count"
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
    colorbar_title: str | None = DEFAULT_COLOR_LEGEND_TITLE,
    size_title: str | None = DEFAULT_SIZE_LEGEND_TITLE,
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
    cmap: Colormap | str | None = DEFAULT_COLORMAP,
    dot_max: float | None = DEFAULT_DOT_MAX,
    dot_min: float | None = DEFAULT_DOT_MIN,
    smallest_dot: float = DEFAULT_SMALLEST_DOT,
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
    dot_color, dot_size = prepare_dataframes_for_dotplot(df)

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
