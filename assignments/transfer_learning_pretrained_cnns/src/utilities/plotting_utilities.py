import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import History

from .logging_utilities import get_logger

logger = get_logger(__name__)


def plot_history(
    H: History,
    num_of_epochs: int,
    save_plot: bool = False,
    output_dir: str = "out",
    plot_name: str = "VGG16_tobacco_plot",
    plot_format: str = "png",
):
    """
    Plots the loss and accuracy curves for a given training history.

    Parameters:
        H (History): The training history object containing the loss and accuracy values.
        num_of_epochs (int): The number of epochs for which the training history is available.
        save_plot (bool, optional): Whether to save the plot as an image file. Defaults to False.
        output_dir (str, optional): The directory where the plot will be saved. Defaults to "out".
        plot_name (str, optional): The name of the plot file. Defaults to "VGG16_tobacco_plot".
        plot_format (str, optional): The format of the plot file. Defaults to "png".

    Raises:
        ValueError: If the specified plot_format is not supported.
    """

    valid_output_formats = [
        "eps",
        "jpeg",
        "jpg",
        "pdf",
        "pgf",
        "png",
        "ps",
        "raw",
        "rgba",
        "svg",
        "svgz",
        "tif",
        "tiff",
    ]
    if plot_format not in valid_output_formats:
        raise ValueError(f"plot_format must be one of {valid_output_formats}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, num_of_epochs), H.history["loss"], label="train_loss")
    plt.plot(
        np.arange(0, num_of_epochs),
        H.history["val_loss"],
        label="val_loss",
        linestyle=":",
    )
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, num_of_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(
        np.arange(0, num_of_epochs),
        H.history["val_accuracy"],
        label="val_acc",
        linestyle=":",
    )
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join("out", plot_name + "." + plot_format))
        logger.info(f"Plot saved as {plot_name}")

    plt.show()
