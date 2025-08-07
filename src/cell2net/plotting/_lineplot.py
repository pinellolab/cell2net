"""Plot functions for prediction module"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
import scanpy as sc
from scanpy.plotting._utils import (
    savefig_or_show,
)

from cell2net.prediction.model import Cell2Net


def train_history(
    model: Cell2Net,
    figsize: tuple[float, float] | None = (10, 4),
    show: bool | None = True,
    save: str | bool | None = None,
    save_prefix: str = "train_history_",
    return_fig: bool | None = False,
) -> None | Figure:
    """
    Lots the training history of a `Cell2Net` model, displaying loss and correlation metrics over epochs.

    Parameters
    ----------
    model :
        A trained `Cell2Net` model with a recorded training history.
    figsize :
        The figure size for the plots, default is (10, 4).
    show :
        Whether to display the plot.
    save :
        If a string is provided, the figure is saved with this filename.
        If True, the figure is saved with the default filename.
        If False or None, the figure is not saved.
    save_prefix :
        Prefix for the filename if `save` is True.
    return_fig :
        If True, returns the figure object instead of displaying or saving it.

    Returns
    -------
        Returns the figure object if `return_fig` is True, otherwise returns None

    Raises
    ------
    AssertionError
        If the model is not trained.

    Notes
    -----
        - The function assumes `model.history` contains `epochs`, `train_loss`, `valid_loss`, `train_corr`, and `valid_corr`.
        - Loss is measured using Negative Log Likelihood.
        - Correlation indicates performance of the model.
        - Uses `savefig_or_show` for handling display or saving of the plot.

    Examples
    --------
    >>> import cell2net as cn
    >>> model = Cell2Net()
    >>> model.train(...)
    >>> cn.pl.train_history(model, show=True)
    """
    # check if model is trained
    assert model.is_trained, print("Please first train the model!")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot train_loss and valid_loss against epoches
    ax1.plot(model.history["epochs"], model.history["train_loss"], label="Train", marker="o")  # type: ignore
    ax1.plot(model.history["epochs"], model.history["valid_loss"], label="Validation", marker="x")  # type: ignore
    ax2.plot(model.history["epochs"], model.history["train_corr"], label="Train", marker="o")  # type: ignore
    ax2.plot(model.history["epochs"], model.history["valid_corr"], label="Validation", marker="x")  # type: ignore

    # Add labels and title
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Negative Log Likelihood")
    ax1.set_title("Train vs Validation")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Correlation")
    ax2.set_title("Train vs Validation")

    ax1.legend()
    ax2.legend()

    fig.tight_layout()

    if return_fig:
        return fig
    else:
        savefig_or_show(save_prefix, show=show, save=save)


def tf_footprint(
    df: pd.DataFrame,
    name: str | None = None,
    palette: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool | None = True,
    save: str | bool | None = None,
    save_dir: str | None = None,
    save_prefix: str = "footprint",
    return_fig: bool | None = False,
) -> None | Figure:
    """
    Plot the transcription factor footprint.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the signal data.
    name : str | None, optional
        Name of the transcription factor, used in the plot title.
    palette : list[str] | None, optional
        Color palette for the plot, by default None.
    figsize : tuple[float, float] | None, optional
        Figure size, by default None
    show : bool | None, optional
        Whether to show the plot, by default True
    save : str | bool | None, optional
        If a string is provided, the figure is saved with this filename.
        If True, the figure is saved with the default filename.
        If False or None, the figure is not saved.
    save_prefix : str, optional
        Prefix for the saved figure filename, by default "tf_footprint_"
    return_fig : bool | None, optional
        If True, returns the figure object instead of displaying or saving it.

    Returns
    -------
    None | Figure
        Returns the figure object if `return_fig` is True, otherwise returns None.
    """

    if save_dir:
        sc.settings.figdir = save_dir

    save_prefix = f"{name}_{save_prefix}" if name else save_prefix

    if figsize is None:
        figsize = (5, 3)

    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=df, x="position", y="signal", hue="label", ax=ax, palette=palette)
    # Set labels and title
    ax.set_xlabel("Position")
    ax.set_ylabel("Signal")
    ax.set_title(f"{name} Footprint" if name else "Footprint")

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        savefig_or_show(save_prefix, show=show, save=save)
