from pathlib import Path

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .logging_utilities import get_logger

logger = get_logger(__name__)


def augment_training_data(use_augmentation=True):
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

def load_saved_model(model_path):
    logger.info(f"Loading model from {model_path}")
    return load_model(model_path)


def instantiate_VGG16_model(
    model_include_top=False,
    model_pooling="avg",
    model_input_shape=(224, 224, 3),
    disable_conv_training=True,
):
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
):
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
    model, output_dir: Path, model_format="keras", file_name="VGG16_tobacco_model"
):
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
    optimizer_type="Adam", init_learn_rate=0.001, decay_steps=10000, decay_rate=0.9
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


def compile_model(model, optimizer):
    # Keras compile modifies model in-place. Return for explicitness.
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
