from pathlib import Path

import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam, SGD, Optimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .logging_utilities import get_logger

logger = get_logger(__name__)

def load_model_history_as_dict(input_directory: Path, file_name: str) -> dict:
    """
    Load the training history from a file.

    Parameters:
        input_directory (Path): The directory where the history file is saved.
        file_name (str): The name of the history file.

    Returns:
        dict: The training history as a dictionary.
    """
    try:
        hist_df = pd.read_csv(input_directory / f"{file_name}.csv")
        number_of_rows = hist_df.shape[0]
        logger.info(f"Model history loaded from {file_name}.csv")
        return hist_df.to_dict('list'), number_of_rows
    except Exception as e:
        logger.error(f"Error in load_model_history_as_dict: {e}")
        return None
    
def save_model_history(H: History, output_directory: Path, file_name: str) -> None:
    """
    Save the training history to a file.

    Parameters:
        H (History): The training history object.
        output_directory (Path): The directory where the history file will be saved.
    """
    hist_df = pd.DataFrame(H.history)
    logger.info(f"Model history saved as {file_name}")
    hist_df.to_csv(output_directory / f"{file_name}.csv", index=False)

def augment_training_data(use_augmentation: bool = True) -> ImageDataGenerator:
    """
    Create an ImageDataGenerator object for augmenting training data.

    Parameters:
    - use_augmentation (bool): Flag indicating whether to use data augmentation. Default is True.

    Returns:
    - datagen (ImageDataGenerator): An ImageDataGenerator object for augmenting training data.
    """
    if use_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=20,
            fill_mode="nearest",
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            validation_split=0.1,
        )
    else:
        # does not modify data, ensures that the data is in the correct format for the model
        datagen = ImageDataGenerator()

    return datagen


def load_saved_model(model_path: Path) -> Model:
    """
    Loads a saved model from the specified path.

    Parameters:
        model_path (str): The path to the saved model file.

    Returns:
        keras.models.Model: The loaded model.
    """
    logger.info(f"Loading model from {model_path}")
    return load_model(model_path)


def instantiate_VGG16_model(
    model_include_top: bool = False,
    model_pooling: str = "avg",
    model_input_shape: tuple = (224, 224, 3),
    disable_conv_training: bool = True,
) -> Model:
    """
    Instantiate a VGG16 model with customizable options. By default, the model is intended for transfer learning through feature extraction.

    Parameters:
        model_include_top (bool, optional): Whether to include the fully-connected
            layer at the top of the network. Defaults to False.
        model_pooling (str, optional): Type of pooling to apply after the last
            convolutional layer. Defaults to "avg".
        model_input_shape (tuple, optional): Shape of the input images. Defaults
            to (224, 224, 3).
        disable_conv_training (bool, optional): Whether to disable training of
            convolutional layers. Defaults to True.

    Returns:
        Model: The instantiated VGG16 model.

    """
    model = VGG16(
        include_top=model_include_top,
        pooling=model_pooling,
        input_shape=model_input_shape,
    )

    if disable_conv_training:
        for layer in model.layers:
            layer.trainable = False

    return model


def define_classification_layers(
    model, batch_normalization_layer=True, dropout_layer=True, print_summary=False
) -> Model:
    """
    Defines the classification layers for a given model.

    Parameters:
        model (Model): The base model to which the classification layers will be added.
        batch_normalization_layer (bool, optional): Whether to include a batch normalization layer. Defaults to True.
        dropout_layer (bool, optional): Whether to include a dropout layer. Defaults to True.
        print_summary (bool, optional): Whether to print the summary of the model. Defaults to False.

    Returns:
        Model: The model with the classification layers added.
    """
    logger.info("Adding classification layers to the model.")
    # Flatten the output of the VGG16 model
    flat1 = Flatten()(model.layers[-1].output)
    # Optionally apply batch normalization
    bn = BatchNormalization()(flat1) if batch_normalization_layer else flat1
    # Activation layer with 128 neurons, relu activation.
    class1 = Dense(128, activation="relu")(bn)
    # Optionally apply dropout
    dropout = Dropout(0.1)(class1) if dropout_layer else class1
    # Output layer neurons, softmax activation.
    output = Dense(10, activation="softmax")(dropout)
    model = Model(inputs=model.inputs, outputs=output)

    if print_summary:
        logger.info(model.summary())

    logger.info("Layers added successfully.")

    return model


def save_trained_model(
    model: Model,
    output_dir: Path,
    model_format: str = "keras",
    file_name: str = "VGG16_tobacco_model",
):
    """
    Save a trained model to the specified output directory.

    Parameters:
        model: The trained model to be saved.
        output_dir (Path): The directory where the model will be saved.
        model_format (str, optional): The format in which the model will be saved.
            Valid options are "HDF5", "SavedModel", and "keras". Defaults to "keras".
        file_name (str, optional): The name of the saved model file. Defaults to "VGG16_tobacco_model".

    Raises:
        ValueError: If an invalid model format is provided.

    Returns:
        None
    """
    valid_formats = ["HDF5", "SavedModel", "keras"]
    if model_format not in valid_formats:
        raise ValueError(
            f"Invalid model format: {model_format}. Expected one of: {valid_formats}"
        )

    if model_format == "HDF5":
        file_name += ".h5"
    elif model_format == "SavedModel":
        file_name += ".pb"
    elif model_format == "keras":
        file_name += ".keras"

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / file_name
    model.save(file_path)
    logger.info(f"Model saved as {file_name}")


def instantiate_optimizer(
    optimizer_type: str = "Adam",
    init_learn_rate: float = 0.001,
    decay_steps: int = 10000,
    decay_rate: float = 0.9,
):
    lr_schedule = ExponentialDecay(
        initial_learning_rate=init_learn_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
    )

    if optimizer_type == "SGD":
        return SGD(learning_rate=lr_schedule)
    elif optimizer_type == "Adam":
        return Adam(learning_rate=lr_schedule)
    else:
        raise ValueError("[ERROR] Invalid optimizer_type. Expected 'SGD' or 'Adam'.")


def compile_model(model: Model, optimizer: Optimizer) -> Model:
    # Keras compile modifies model in-place. Return for explicitness.
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
