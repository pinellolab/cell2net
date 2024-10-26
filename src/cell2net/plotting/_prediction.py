"""Plot functions for prediction module"""

import matplotlib.pyplot as plt

from cell2net.prediction.model import Cell2Net


def train_history(
    model: Cell2Net, figsize: tuple[float, float] | None = (10, 4), show: bool = True
):
    # check if model is trained
    assert model.is_trained, print("Please first train the model!")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot train_loss and valid_loss against epoches
    ax1.plot(model.history["epochs"], model.history["train_loss"], label="Train", marker="o")  # type: ignore
    ax1.plot(model.history["epochs"], model.history["valid_loss"], label="Validation", marker="x")  # type: ignore
    ax2.plot(model.history["epochs"], model.history["train_corr"], label="Train", marker="o")  # type: ignore
    ax2.plot(model.history["epochs"], model.history["valid_corr"], label="Validation", marker="x")  # type: ignore

    # Add labels and title
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train vs Validation")

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Correlation")
    ax2.set_title("Train vs Validation")

    ax1.legend()
    ax2.legend()

    plt.show()
