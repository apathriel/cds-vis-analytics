from pathlib import Path

import click
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report


from utilities.data_processing_utilities import (
    binarize_and_fit_labels,
    get_unique_labels_from_subdirs,
    initialize_data_directory,
    load_and_preprocess_training_data,
    split_data,
)
from utilities.logging_utilities import get_logger

from utilities.model_compilation_utilities import (
    augment_training_data,
    compile_model,
    define_classification_layers,
    instantiate_optimizer,
    instantiate_VGG16_model,
    load_saved_model,
    save_trained_model,
    save_model_history,
    load_model_history_as_dict
)
from utilities.plotting_utilities import plot_history

# Logging setup
logger = get_logger(__name__)


def model_pipeline(
    model_directory: Path = None,
    model_file_name: str = None,
    load_existing_model: bool = False,
    output_model_summary: bool = False,
    optimizer_type: str = "Adam",
) -> Model:
    """
    Function responsible for generating and compiling a model architecture based on pre-trained model for document classification.

    Parameters:
        model_directory (Path): The directory where the model file is located.
        model_file_name (str): The name of the model file.
        load_existing_model (bool): Whether to load an existing model from the given model path. Default is False.
        output_model_summary (bool): Whether to print the summary of the compiled model. Default is False.
        optimizer_type (str): The type of optimizer to use for model compilation. Written for either Adam or SGD. Default is Adam.

    Returns:
        compiled_model (Model): The compiled model for document classification.
    """
    if load_existing_model:
        model_path = model_directory / model_file_name
        logger.info(f"Loading model from {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")
        compiled_model = load_saved_model(model_path)
    else:
        logger.info("Creating new model...")
        optimizer = instantiate_optimizer(optimizer_type=optimizer_type)
        model = define_classification_layers(instantiate_VGG16_model())
        compiled_model = compile_model(model, optimizer)

    if output_model_summary:
        logger.info(compiled_model.summary())

    return compiled_model

def save_classification_report(
    classification_report: str,
    output_dir: Path,
    log_output: bool = True,
    file_name: str = "VGG16_tobacco_report.txt",
) -> None:
    """
    Saves the classification report to a file.

    Paramters:
        classification_report (str): The classification report to be saved.
        output_dir (Path): The directory where the file will be saved.
        log_output (bool, optional): Whether to log the classification report. Defaults to True.
        file_name (str, optional): The name of the file to be saved. Defaults to "VGG16_tobacco_report.txt".
    """
    if log_output:
        logger.info(f"Classification Report:\n{classification_report}")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / file_name
    with open(file_path, "w") as file:
        file.write(classification_report)
    logger.info(f"Classification report saved as {file_name}")


@click.command()
@click.option(
    "--num_of_epochs",
    "-e",
    default=12,
    type=int,
    help="Number of epochs to train the model",
)
@click.option(
    "--batch_size",
    "-b",
    default=128,
    type=int,
    help="Batch size for training the model",
)
@click.option(
    "--optimizer_type",
    "-o",
    default="Adam",
    help="Type of optimizer to use. Script is written to accept either SGD or Adam. Default is Adam.",
)
@click.option(
    "--test_split_size",
    "-t",
    default=0.2,
    type=float,
    help="The proportion of the data to use for testing the model. Default is 0.2",
)
@click.option(
    "--print_model_summary",
    "-p",
    is_flag=True,
    default=False,
    type=bool,
    help="Flag determining whether to print the model summary",
)
@click.option(
    "--use_saved_model",
    "-s",
    is_flag=True,
    default=False,
    type=bool,
    help="Flag determining whether to use a saved model",
)
@click.option(
    "--model_title",
    "-m",
    default="VGG16_tobacco_model.keras",
    type=str,
    help="The file name of the saved model. Default is VGG16_tobacco_model.keras. Must be placed in out/models directory.",
)
def main(
    num_of_epochs: int,
    batch_size: int,
    optimizer_type: str,
    test_split_size: float,
    print_model_summary: bool,
    use_saved_model: bool,
    model_title: str,
):
    # Instantiate directory paths
    path_to_input_directory = Path(__file__).parent / ".." / "in"
    path_to_output_directory = Path(__file__).parent / ".." / "out"
    path_to_model_directory = path_to_output_directory / "models"

    # Initialize data directory
    data_dir = initialize_data_directory(path_to_input_directory)

    # Instantiate model architecture
    model = model_pipeline(
        model_directory=path_to_model_directory,
        model_file_name=model_title,
        load_existing_model=use_saved_model,
        optimizer_type=optimizer_type,
        output_model_summary=print_model_summary,
    )

    # Load and preprocess training data, split data, binarize labels
    X, y = load_and_preprocess_training_data(data_dir)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_split_size, validation_size=None, stratify=y
    )
    y_train, y_test = binarize_and_fit_labels(y_train, y_test)
    data_gen = augment_training_data(use_augmentation=True)

    if use_saved_model:
         H, number_of_rows = load_model_history_as_dict(path_to_model_directory, "VGG16_tobacco_history")
    else:
        number_of_rows = None
        # Augment training data, will only modify training data if use_augmentation is True
       
        # Early stopping functionality
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, verbose=1, mode="auto"
        )
        # Fit model to training data
        logger.info("Starting model training.")
        H = model.fit(
            data_gen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=data_gen.flow(
                X_train, y_train, batch_size=batch_size, subset="validation"
            ),
            epochs=num_of_epochs,
            verbose=1,
            callbacks=[early_stopping],
        )
        logger.info("Model training completed.")

        save_model_history(H, path_to_model_directory, file_name="VGG16_tobacco_history")
        save_trained_model(model, path_to_model_directory, model_format="keras")

    predictions = model.predict(X_test, batch_size=batch_size)
    labels = get_unique_labels_from_subdirs(data_dir)

    VGG16_report = classification_report(
        y_test.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=labels,
        zero_division=1,
    )

    save_classification_report(
        VGG16_report, path_to_output_directory, file_name="VGG16_tobacco_report.txt"
    )

    plot_history(
        H,
        num_of_csv_rows=number_of_rows,
        save_plot=True,
        output_dir="out",
        plot_name="VGG16_tobacco_plot",
        plot_format="pdf",
    )


if __name__ == "__main__":
    main()
