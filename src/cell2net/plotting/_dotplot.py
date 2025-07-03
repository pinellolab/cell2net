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
from scanpy.plotting._baseplot_class import BasePlot, VBoundNorm
from scanpy.plotting._utils import (
    ColorLike,
    _AxesSubplot,
    check_colornorm,
    fix_kwds,
    make_grid_spec,
    savefig_or_show,
)

from cell2net._logging import logger

_VarNames = str | Sequence[str]

if TYPE_CHECKING:
    from typing import Literal, Self

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

# default style parameters
DEFAULT_COLORMAP = "Reds"
DEFAULT_COLOR_ON = "dot"
DEFAULT_DOT_MAX = 200
DEFAULT_DOT_MIN = 0
DEFAULT_SMALLEST_DOT = 0.0
DEFAULT_LARGEST_DOT = 200.0
DEFAULT_DOT_EDGECOLOR = "black"
DEFAULT_DOT_EDGELW = 0.2

# default legend parameters
DEFAULT_SIZE_LEGEND_TITLE = "Number of target \ngenes in group"
DEFAULT_COLOR_LEGEND_TITLE = "Mean regulation \nin group"
DEFAULT_LEGENDS_WIDTH = 1.5  # inches
DEFAULT_PLOT_X_PADDING = 0.8  # a unit is the distance between two x-axis ticks
DEFAULT_PLOT_Y_PADDING = 1.0  # a unit is the distance between two y-axis ticks

# gridspec parameter. Sets the space between mainplot, dendrogram and legend
DEFAULT_WSPACE = 0

MIN_FIGURE_HEIGHT = 2.5


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
        var_names: _VarNames | Mapping[str, _VarNames] | None,
        standard_scale: Literal["var", "group"],
        var_group_positions: Sequence[tuple[int, int]] | None = None,
        var_group_labels: Sequence[str] | None = None,
        var_group_rotation: float | None = None,
        categories_order: Sequence[str] | None = None,
        title: str | None = None,
        colorbar_title: str = "",
        size_title: str = "",
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

        self.has_var_groups = True if var_group_positions is not None and len(var_group_positions) > 0 else False

        self.categories_order = categories_order

        self.var_groups = None
        # self.var_names, self.var_groups = _var_groups(var_names)

        self._update_var_groups()

        dot_color_df = dot_color_df.loc[:, self.var_names]  # type: ignore
        dot_size_df = dot_size_df.loc[:, self.var_names]  # type: ignore

        # Because genes (columns) can be duplicated (e.g. when the
        # same gene is reported as marker gene in two clusters)
        # they need to be removed first,
        # otherwise, the duplicated genes are further duplicated when reordering
        # Eg. A df with columns ['a', 'b', 'a'] after reordering columns
        # with df[['a', 'a', 'b']], results in a df with columns:
        # ['a', 'a', 'a', 'a', 'b']

        unique_var_names, unique_idx = np.unique(dot_color_df.columns, return_index=True)

        # remove duplicate columns
        if len(unique_var_names) != len(self.var_names):  # type: ignore
            dot_color_df = dot_color_df.iloc[:, unique_idx]

        # get the same order for rows and columns in the dot_color_df
        # using the order from the doc_size_df
        dot_color_df = dot_color_df.loc[dot_size_df.index][dot_size_df.columns]

        self.categories = dot_color_df.index.tolist()

        self.dot_color_df, self.dot_size_df = (
            df.loc[categories_order if categories_order is not None else self.categories]  # type: ignore
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

        self.kwds = kwds
        self.vboundnorm = VBoundNorm(vmin=vmin, vmax=vmax, vcenter=vcenter, norm=norm)

        # Set fig title
        self.fig_title = title

        # Set default style parameters
        self.cmap = DEFAULT_COLORMAP
        self.dot_max = DEFAULT_DOT_MAX
        self.dot_min = DEFAULT_DOT_MIN
        self.smallest_dot = DEFAULT_SMALLEST_DOT
        self.largest_dot = DEFAULT_LARGEST_DOT
        self.color_on = DEFAULT_COLOR_ON
        self.grid = False
        self.plot_x_padding = DEFAULT_PLOT_X_PADDING
        self.plot_y_padding = DEFAULT_PLOT_Y_PADDING

        self.dot_edge_color = DEFAULT_DOT_EDGECOLOR
        self.dot_edge_lw = DEFAULT_DOT_EDGELW

        # style default parameters
        self.are_axes_swapped = False
        self.categories_order = categories_order
        self.var_names_idx_order = None

        # set default values for legend
        self.color_legend_title = colorbar_title
        self.size_title = size_title
        self.legends_width = DEFAULT_LEGENDS_WIDTH
        self.show_size_legend = True
        self.show_colorbar = True

        self.wspace = DEFAULT_WSPACE

        self.group_extra_size = 0
        self.plot_group_extra = None

        # minimum height required for legends to plot properly
        self.min_figure_height = MIN_FIGURE_HEIGHT

        # after .render() is called the fig value is assigned and ax_dict
        # contains a dictionary of the axes used in the plot
        self.fig = None
        self.ax_dict = None
        self.ax = ax

    def style(
        self,
        *,
        cmap: Colormap | str,
        color_on: Literal["dot", "square"],
        smallest_dot: float | Empty = _empty,
        largest_dot: float | Empty = _empty,
        dot_max: float | None | Empty = _empty,
        dot_min: float | None | Empty = _empty,
        dot_edge_color: ColorLike | None | Empty = _empty,
        dot_edge_lw: float | None | Empty = _empty,
        grid: bool | Empty = _empty,
        x_padding: float | Empty = _empty,
        y_padding: float | Empty = _empty,
    ) -> Self:
        """
        Modifies plot visual parameters

        Parameters
        ----------
        cmap : Colormap | str | None
            String denoting matplotlib color map
        color_on : Literal["dot", "square"]
            By default the color map is applied to the color of the ``"dot"``.
            Optionally, the colormap can be applied to a ``"square"`` behind the dot,
            in which case the dot is transparent and only the edge is shown, by default _empty
        smallest_dot : float
            Smallest dot size
        largest_dot : float
            Largest dot size
        dot_max : float | None | Empty, optional
            If ``None``, the maximum dot size is set to the maximum number of target genes found.
            If given, the value should be a number between 0 and 1.
            All fractions larger than dot_max are clipped to this value, by default _empty
        dot_min : float | None | Empty, optional
            If ``None``, the minimum dot size is set to 0.
            If given, the value should be a number between 0 and 1.
            All fractions smaller than dot_min are clipped to this value, by default _empty
        dot_edge_color : ColorLike | None | Empty, optional
            Dot edge color.
            When `color_on='dot'`, ``None`` means no edge.
            When `color_on='square'`, ``None`` means that
            the edge color is white for darker colors and black for lighter background square, by default _empty
        dot_edge_lw : float | None | Empty, optional
            Dot edge line width.
            When `color_on='dot'`, ``None`` means no edge.
            When `color_on='square'`, ``None`` means a line width of 1.5, by default _empty
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

        return self

    def _plot_size_legend(self, size_legend_ax: Axes):
        # for the dot size legend, use step between dot_max and dot_min
        # based on how different they are.
        diff = self.dot_max - self.dot_min  # type: ignore
        if diff <= 30:
            step = 5
        elif 30 < diff <= 60:
            step = 10
        elif 60 < diff <= 100:
            step = 20
        elif 100 < diff <= 200:
            step = 50
        elif 200 < diff <= 300:
            step = 100
        else:
            step = 200

        # a descending range that is afterwards inverted is used
        # to guarantee that dot_max is in the legend.
        size_range = np.arange(self.dot_max, self.dot_min, step * -1)[::-1]
        dot_range = self.dot_max - self.dot_min  # type: ignore
        size_values = (size_range - self.dot_min) / dot_range
        size = size_values * (self.largest_dot - self.smallest_dot) + self.smallest_dot

        # plot size bar
        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            color="gray",
            edgecolor="black",
            linewidth=self.dot_edge_lw,
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = [f"{np.round((x), decimals=0).astype(int)}" for x in size_range]
        size_legend_ax.set_xticklabels(labels, fontsize="small")

        # remove y ticks and labels
        size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)

        # remove surrounding lines
        size_legend_ax.spines["right"].set_visible(False)
        size_legend_ax.spines["top"].set_visible(False)
        size_legend_ax.spines["left"].set_visible(False)
        size_legend_ax.spines["bottom"].set_visible(False)
        size_legend_ax.grid(visible=False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size="small")

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    def _plot_legend(self, legend_ax, return_ax_dict, normalize):
        # to maintain the fixed height size of the legends, a
        # spacer of variable height is added at the bottom.
        # The structure for the legends is:
        # first row: variable space to keep the other rows of
        #            the same size (avoid stretching)
        # second row: legend for dot size
        # third row: spacer to avoid color and size legend titles to overlap
        # fourth row: colorbar

        cbar_legend_height = self.min_figure_height * 0.08
        size_legend_height = self.min_figure_height * 0.27
        spacer_height = self.min_figure_height * 0.3

        height_ratios = [
            self.height - size_legend_height - cbar_legend_height - spacer_height,  # type: ignore
            size_legend_height,
            spacer_height,
            cbar_legend_height,
        ]
        fig, legend_gs = make_grid_spec(legend_ax, nrows=4, ncols=1, height_ratios=height_ratios)

        if self.show_size_legend:
            size_legend_ax = fig.add_subplot(legend_gs[1])
            self._plot_size_legend(size_legend_ax)
            return_ax_dict["size_legend_ax"] = size_legend_ax

        if self.show_colorbar:
            color_legend_ax = fig.add_subplot(legend_gs[3])

            self._plot_colorbar(color_legend_ax, normalize)
            return_ax_dict["color_legend_ax"] = color_legend_ax

    def legend(
        self,
        *,
        show: bool | None = True,
        show_size_legend: bool | None = True,
        show_colorbar: bool | None = True,
        size_title: str = DEFAULT_SIZE_LEGEND_TITLE,
        colorbar_title: str = DEFAULT_COLOR_LEGEND_TITLE,
        width: float = DEFAULT_LEGENDS_WIDTH,
    ) -> Self:
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
        Makes a *dot plot* given two data frames

        one containing the doc size and other containing the dot color. The indices and
        columns of the data frame are used to label the output image

        Parameters
        ----------
        dot_size : pd.DataFrame
            _description_
        dot_color : pd.DataFrame
            _description_
        dot_ax : Axes
            _description_
        cmap : Colormap | str | None
            _description_
        color_on : Literal[&quot;dot&quot;, &quot;square&quot;]
            _description_
        dot_max : float | None
            _description_
        dot_min : float | None
            _description_
        standard_scale : Literal[&quot;var&quot;, &quot;group&quot;] | None
            _description_
        smallest_dot : float
            _description_
        largest_dot : float
            _description_
        edge_color : ColorLike | None
            _description_
        edge_lw : float | None
            _description_
        grid : bool
            _description_
        x_padding : float
            _description_
        y_padding : float
            _description_
        vmin : float | None
            _description_
        vmax : float | None
            _description_
        vcenter : float | None
            _description_
        norm : Normalize | None
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        assert dot_size.shape == dot_color.shape, (
            "please check that dot_size " "and dot_color dataframes have the same shape"
        )

        assert list(dot_size.index) == list(dot_color.index), (
            "please check that dot_size " "and dot_color dataframes have the same index"
        )

        assert list(dot_size.columns) == list(dot_color.columns), (
            "please check that the dot_size " "and dot_color dataframes have the same columns"
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
        n_targets = dot_size.values.flatten()  # get number of targets
        mean_flat = dot_color.values.flatten()
        cmap = plt.get_cmap(cmap)  # type: ignore
        if dot_max is None:
            dot_max = np.max(n_targets)
        if dot_min is None:
            dot_min = 0

        n_targets = np.clip(n_targets, dot_min, dot_max)
        old_range = dot_max - dot_min  # type: ignore

        # re-scale targets between 0 and 1
        n_targets = (n_targets - dot_min) / old_range

        # rescale size to match smallest_dot and largest_dot
        size = n_targets * (largest_dot - smallest_dot) + smallest_dot
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
        dot_ax.set_yticklabels([dot_color.index[idx] for idx, _ in enumerate(y_ticks)], minor=False)

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

    def _update_var_groups(self) -> None:
        """
        Checks if var_names is a dict.

        If this is the cases, then set the
        correct values for var_group_labels and var_group_positions

        updates var_names, var_group_labels, var_group_positions
        """
        if isinstance(self.var_names, Mapping):
            if self.has_var_groups:
                logger.warning(
                    "`var_names` is a dictionary. This will reset the current "
                    "values of `var_group_labels` and `var_group_positions`."
                )
            var_group_labels = []
            _var_names = []
            var_group_positions = []
            start = 0
            if self.categories_order:
                for label in self.categories_order:
                    vars_list = self.var_names[label]
                    if isinstance(vars_list, str):
                        vars_list = [vars_list]
                    # use list() in case var_list is a numpy array or pandas series
                    _var_names.extend(list(vars_list))
                    var_group_labels.append(label)
                    var_group_positions.append((start, start + len(vars_list) - 1))
                    start += len(vars_list)
            else:
                for label, vars_list in self.var_names.items():
                    if isinstance(vars_list, str):
                        vars_list = [vars_list]
                    # use list() in case var_list is a numpy array or pandas series
                    _var_names.extend(list(vars_list))
                    var_group_labels.append(label)
                    var_group_positions.append((start, start + len(vars_list) - 1))
                    start += len(vars_list)
            self.var_names = _var_names
            self.var_group_labels = var_group_labels
            self.var_group_positions = var_group_positions
            self.has_var_groups = True

        elif isinstance(self.var_names, str):
            self.var_names = [self.var_names]


def prepare_dataframes_for_dotplot(
    df: pd.DataFrame,
    tf_col: str = "tf",
    gene_col: str = "gene",
    group_col: str = "cell_type",
    activity_col: str = "activity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the color and size dataframe for dot plot

    Parameters
    ----------
    df : pd.DataFrame
        Input data, should be a DataFrame with at least three columns with following format
    tf_col : str, optional
        Column name for TF names, by default "tf"
    gene_col: str, optional
        Column name for gene names, by default "gene"
    group_col : str, optional
        Column name for groups, by default "cell_type"
    activity_col : str, optional
        Column name for regulation activity of tf-gene in a group, by default "activity"

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two dataframes for making dot plot
    """
    # Make sure the column names can be found in dataframe
    assert tf_col in df.columns, f"Cannot find {tf_col} in dataframe"
    assert gene_col in df.columns, f"Cannot find {gene_col} in dataframe"
    assert group_col in df.columns, f"Cannot find {group_col} in dataframe"
    assert activity_col in df.columns, f"Cannot find {activity_col} in dataframe"

    # Get average regulation of each tf within cell types, used for dot color
    dot_color_df = df.groupby([tf_col, group_col])[activity_col].sum().reset_index()
    dot_color_df = dot_color_df.pivot_table(index=group_col, columns=tf_col, values=activity_col)
    dot_color_df = dot_color_df.fillna(0)
    dot_color_df = dot_color_df.rename_axis(None, axis=0)
    dot_color_df = dot_color_df.rename_axis(None, axis=1)

    # Get number of target genes of each within cell types, used for dot size
    dot_size_df = df.groupby([tf_col, group_col])[activity_col].count().reset_index(name="count")
    dot_size_df = dot_size_df.pivot_table(index=group_col, columns=tf_col, values="count")
    dot_size_df = dot_size_df.fillna(0)
    dot_size_df = dot_size_df.rename_axis(None, axis=0)
    dot_size_df = dot_size_df.rename_axis(None, axis=1)

    return dot_color_df, dot_size_df


def tf_dotplot(
    df: pd.DataFrame,
    tf_col: str = "tf",
    gene_col: str = "gene",
    group_col: str = "group",
    activity_col: str = "activity",
    var_names: _VarNames | Mapping[str, _VarNames] | None = None,
    categories_order: Sequence[str] | None = None,
    standard_scale: Literal["var", "group"] = "var",
    title: str = "",
    colorbar_title: str = "Mean regulation \nin group",
    size_title: str = "Number of target \ngenes in group",
    figsize: tuple[float, float] | None = None,
    swap_axes: bool | None = False,
    show: bool | None = None,
    save: str | bool | None = None,
    save_prefix: str = "dotplot_",
    ax: _AxesSubplot | None = None,
    return_fig: bool | None = False,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    # Style parameters
    cmap: Colormap | str = "Reds",
    color_on: Literal["dot", "square"] = "dot",
    n_targets_max: float = 200,
    n_targets_min: float = 0,
    largest_dot: float = 200,
    smallest_dot: float = 0,
    **kwds,
) -> DotPlot | dict | None:
    """
    Makes a dot plot of the regulation activities of `var_names`.

    This function is modified from sc.pl.dotplot and provides the same interface
    to visualize TF regulation for each group which can be different cell
    types or pseudotime point in along dynamic process.

    Each dot represents two values: mean regulation activity within each category
    (visualized by color) and number of target genes of the `var_name` in the
    category (visualized by the size of the dot).

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe containing the main input for the dot plot.
        This dataframe must contain at least four columns:
        +---------+-----------+-----------+----------+
        | tf      | gene      |    group  | activity |
        | TF1     | Gene1     | Celltype1 |  0.32    |
        | TF2     | Gene1     | Celltype1 |  0.42    |
        | TF1     | Gene2     | Celltype2 |  0.12    |
        +---------+-----------+-----------+----------+
    tf_col :
        Column name for TF names, by default "tf"
    gene_col:
        Column name for gene names, by default "gene"
    group_col :
        Column name for groups, by default "group"
    activity_col :
        Column name for regulation activity of tf-gene in a group, by default "activity"
    var_names:

    Returns
    -------
    DotPlot or dict or None
        If `return_fig` is True, returns the `DotPlot` object.
        Otherwise, generates the figure and either displays or saves it.
    """
    # prepare data for plotting
    dot_color, dot_size = prepare_dataframes_for_dotplot(
        df=df,
        tf_col=tf_col,
        gene_col=gene_col,
        group_col=group_col,
        activity_col=activity_col,
    )

    dp = DotPlot(
        dot_color_df=dot_color,
        dot_size_df=dot_size,
        var_names=var_names,
        categories_order=categories_order,
        standard_scale=standard_scale,
        title=title,
        colorbar_title=colorbar_title,
        size_title=size_title,
        figsize=figsize,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
    )

    if swap_axes:
        dp.swap_axes()

    dp = dp.style(
        cmap=cmap,
        color_on=color_on,
        dot_max=n_targets_max,
        dot_min=n_targets_min,
        largest_dot=largest_dot,
        smallest_dot=smallest_dot,
        dot_edge_lw=kwds.pop("linewidth", _empty),
    ).legend(colorbar_title=colorbar_title, size_title=size_title)  # type: ignore

    if return_fig:
        return dp
    else:
        dp.make_figure()
        savefig_or_show(save_prefix, show=show, save=save)
        # show = settings.autoshow if show is None else show
        # if not show:
        #     return dp.get_axes()
