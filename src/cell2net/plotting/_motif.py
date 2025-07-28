from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import pyranges as pr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scanpy.plotting._utils import (
    savefig_or_show,
)

from cell2net._logging import logger

_COLOR_SCHEMES = Literal["classic", "grays", "base_pairing", "colorblind_safe"]


def motif_logo(
    motifs: Iterable,
    motif_name: str,
    color_scheme: str | _COLOR_SCHEMES = "classic",
    figsize: tuple[float, float] | None = None,
    show: bool | None = True,
    save: str | bool | None = None,
    save_prefix: str = "motif_logo_",
    return_fig: bool | None = False,
) -> None | Figure:
    """
    Generate a sequence motif logo from a count matrix.

    This function creates a sequence motif logo based on the given count matrix,
    using information content to represent sequence conservation.
    The logo is generated using the `logomaker` library.

    Parameters
    ----------
    motifs :
        An iterable of motif objects, each containing a `counts` attribute.
        The `counts` attribute should be a dictionary where keys represent sequence characters
        (e.g., 'A', 'C', 'G', 'T') and values are lists of counts across motif positions.
    motif_name :
        The name of the motif to be visualized.
    color_scheme :
        The color scheme used for visualization, by default "classic".
    figsize :
        Figure size (width, height) in inches, by default None, which defaults to (5, 3).
    show :
        Whether to display the plot.
    save :
        If a string is provided, the figure is saved with this filename.
        If True, the figure is saved with the default filename.
        If False or None, the figure is not saved.
    save_prefix :
        Prefix for the saved figure filename.
    return_fig :
        If True, returns the figure object instead of displaying or saving it.

    Returns
    -------
    If `return_fig` is True, returns the `matplotlib.figure.Figure` object. Otherwise, the function displays or saves the figure and returns None.

    Notes
    -----
    - The function computes the information content matrix (ICM) from the PWM by normalizing counts, computing log probabilities, and scaling by information content.
    - Requires `logomaker` for visualization; raises an error if not installed.
    - The figure is styled to remove spines and set a fixed y-axis range of [0, 2].

    Examples
    --------
    >>> import cell2net as cn
    >>> counts = {"A": [10, 2, 3, 4], "C": [1, 5, 0, 7], "G": [4, 1, 6, 2], "T": [2, 8, 1, 3]}
    >>> cn.pl.motif_logo(counts, color_scheme="classic", show=True)
    """
    try:
        import logomaker
    except ImportError:
        logger.error(
            "logomaker is not installed. Please install it with: pip install logomaker"
        )
        return None

    # get motif counts
    for motif in motifs:
        if motif_name.upper() == motif.name.upper():
            counts = motif.counts
            break
    else:
        logger.error(f"Motif {motif_name} not found in the motifs list.")
        return None

    # compute information content matrix
    pwm = pd.DataFrame(data=counts)
    # add a small value to avoid division by zero
    pwm = pwm.add(0.0001)
    pwm_prob = (pwm.T / pwm.T.sum()).T
    pwm_prob_log = np.log2(pwm_prob)
    pwm_prob_log = pwm_prob_log * pwm_prob
    info_content = pwm_prob_log.T.sum() + 2
    icm = pwm_prob.mul(info_content, axis=0)

    # create logo
    if figsize is None:
        figsize = (5, 3)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    logo = logomaker.Logo(icm, ax=ax, color_scheme=color_scheme, baseline_width=0)
    logo.style_spines(visible=False)
    ax.set_ylim(ymin=0, ymax=2)
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel("")
    fig.tight_layout()

    if return_fig:
        return fig
    else:
        savefig_or_show(save_prefix, show=show, save=save)

