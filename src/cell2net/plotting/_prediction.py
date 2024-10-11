"""Plot functions for prediction module"""

import matplotlib.pyplot as plt

from cell2net.prediction.model import Cell2Net


def train_history(
    model: Cell2Net, figsize: tuple[float, float] | None = (4, 4), show: bool = True
):
    # check if model is trained

    assert model.is_trained, print("Model hasn't been trained!")  # type: ignore

    df = model.history

    _, ax = plt.subplots(figsize=figsize)

    # Plot train_loss and valid_loss against epoches
    ax.plot(df["epochs"], df["train_loss"], label="Train Loss", marker="o")
    ax.plot(df["epochs"], df["valid_loss"], label="Validation Loss", marker="x")

    # Add labels and title
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")

    ax.legend()

    if show:
        plt.show()
    else:
        return ax
