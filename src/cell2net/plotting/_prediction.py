"""Plot functions for prediction module"""

import matplotlib.pyplot as plt

from cell2net.prediction.model import Cell2Net


def train_history(
    model: Cell2Net, figsize: tuple[float, float] | None = (4, 4), show: bool = True
):
    # check if model is trained
    assert model.is_trained, print("Please first train the model!")

    _, ax = plt.subplots(figsize=figsize)

    # Plot train_loss and valid_loss against epoches
    ax.plot(model.history["epochs"], model.history["train_loss"], label="Train", marker="o")  # type: ignore
    ax.plot(model.history["epochs"], model.history["valid_loss"], label="Validation", marker="x")  # type: ignore

    # Add labels and title
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")

    ax.legend()

    if show:
        plt.show()
    else:
        return ax
