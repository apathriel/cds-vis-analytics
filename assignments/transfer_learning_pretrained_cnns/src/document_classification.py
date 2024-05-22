from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report


from data_processing_utilities import (
    binarize_and_fit_labels,
    get_unique_labels_from_subdirs,
    initialize_data_directory,
    load_and_preprocess_training_data,
    split_data,
)
from logger_utils import get_logger

from model_compilation_utilities import (
    compile_model,
    define_classification_layers,
    instantiate_optimizer,
    instantiate_VGG16_model,
    load_saved_model,
    save_trained_model,
)
from plotting_utilities import plot_history

# Logging setup
logger = get_logger(__name__)


def model_pipeline(model_path: Path = None, load_existing_model: bool = False, output_model_summary: bool = False, optimizer_type: str = "Adam"):
    if load_existing_model:
        logger.info(f"Loading model from {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")
        return load_saved_model(model_path)
    
    optimizer = instantiate_optimizer(optimizer_type=optimizer_type)
    model = define_classification_layers(instantiate_VGG16_model())
    compiled_model = compile_model(model, optimizer)
    return compiled_model


def augment_training_data(X_train, y_train, use_augmentation=True):
    if use_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=20,
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            fill_mode="nearest",
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            validation_split=0.1,
        )
    else:
        # does not modify data, ensures that the data is in the correct format for the model
        datagen = ImageDataGenerator()

    data_gen = datagen.flow(X_train, y_train, batch_size=128)

    return data_gen



def save_classification_report(classification_report: str, output_dir: Path, log_output: bool =True, file_name: str = "VGG16_tobacco_report.txt"):
    if log_output:
        logger.info(f"Classification Report:\n{classification_report}")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / file_name
    with open(file_path, "w") as file:
        file.write(classification_report)
    logger.info(f"Classification report saved as {file_name}")


if __name__ == "__main__":
    # Instantiate directory paths
    path_to_input_directory = Path(__file__).parent / ".." / "in"
    path_to_output_directory = Path(__file__).parent / ".." / "out"
    path_to_model_directory = path_to_output_directory / "models"

    # Initialize data directory
    data_dir = initialize_data_directory(path_to_input_directory)

    model = model_pipeline(optimizer_type="Adam")

    X, y = load_and_preprocess_training_data(data_dir)
    print(X.shape, y.shape)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    y_train, y_val, y_test = binarize_and_fit_labels(y_train, y_val, y_test)

    data_gen = augment_training_data(X_train, y_train, use_augmentation=True)

    logger.info("Starting model training.")
    H = model.fit(
        data_gen, validation_data=(X_val, y_val), batch_size=128, epochs=10, verbose=1
    )
    logger.info("Model training completed.")

    save_trained_model(model, path_to_model_directory, model_format="keras")

    predictions = model.predict(X_test, batch_size=128)
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
        10,
        save_plot=True,
        output_dir="out",
        plot_name="VGG16_tobacco_plot",
        plot_format="pdf",
    )
