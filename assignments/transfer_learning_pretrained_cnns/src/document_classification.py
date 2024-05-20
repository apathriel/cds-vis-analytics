import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

from tqdm import tqdm

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

logger.addHandler(ch)


# Data processing, loading and prep
def initialize_data_directory(data_path):
    data_dir = Path(data_path)

    # Check if the directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")

    # Check if the directory is accessible
    if not data_dir.is_dir():
        raise NotADirectoryError(f"The path {data_dir} is not a directory.")

    return data_dir


def create_label_list_from_subdirs(image_files):
    labels = []
    # For each image file, get the parent directory name and use it as label
    for image_file in image_files:
        labels.append(os.path.basename(os.path.dirname(image_file)))
    labels = np.array(labels)
    return labels


def get_unique_labels_from_subdirs(directory):
    return os.listdir(directory)


def load_image_training_data(data_path):
    try:
        logger.info(f"Loading images...")
        # load all jpg files in the data directory, create labels
        image_files = sorted([str(file) for file in Path(data_path).rglob("*.jpg")])
        labels = create_label_list_from_subdirs(image_files)
        # load images using keras/pillow
        loaded_images = [
            load_img(img, color_mode="rgb", target_size=(224, 224))
            for img in tqdm(image_files, desc="Loading images")
        ]
        # convert images to numpy arrays and preprocess for VGG16 model
        preprocessed_images = np.array(
            [
                preprocess_input(img_to_array(img))
                for img in tqdm(loaded_images, desc="Preprocessing images")
            ]
        )
        logger.info(
            f"Loaded {len(preprocessed_images)} images and {len(labels)} labels."
        )
        return preprocessed_images, labels
    except Exception as e:
        logger.error(f"Error in load_image_training_data: {e}")
        return None, None


# Initialize the VGG16 model
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
    model, batch_normalization_layer=True, dropout_layer=False, print_summary=False
):
    logger.info("Adding classification layers to the model.")
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1) if batch_normalization_layer else flat1
    class1 = Dense(128, activation="relu")(bn)
    dropout = Dropout(0.5)(class1) if dropout_layer else class1
    output = Dense(10, activation="softmax")(dropout)
    model = Model(inputs=model.inputs, outputs=output)
    if print_summary:
        logger.info(model.summary())
    logger.info("Layers added successfully.")
    return model


def instantiate_optimizer(
    optimizer_type="Adam", init_learn_rate=0.001, decay_steps=100000, decay_rate=0.9
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
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


def model_pipeline(load_existing_model=False, model_path="", optimizer_type="Adam"):
    model_path = Path(model_path)
    if load_existing_model:
        logger.info(f"Loading model from {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")
        return load_saved_model(model_path)
    optimizer = instantiate_optimizer(optimizer_type=optimizer_type)
    model = define_classification_layers(instantiate_VGG16_model())
    compile_model(model, optimizer)
    return model


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


def split_data(X, y, test_size=0.25, validation_size=0.1):
    # First split to separate out the training set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + validation_size
    )

    # Second  split to separate out the validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + validation_size)
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def binarize_and_fit_labels(y_train, y_val, y_test):
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_val = lb.transform(y_val)
    y_test = lb.transform(y_test)
    return y_train, y_val, y_test


def load_saved_model(model_path):
    logger.info(f"Loading model from {model_path}")
    return load_model(model_path)


def save_trained_model(
    model, output_dir, model_format="keras", file_name="VGG16_tobacco_model"
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

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    model.save(file_path)
    logger.info(f"Model saved as {file_name}")


def plot_history(
    H,
    num_of_epochs: int,
    save_plot: bool = False,
    output_dir: str = "out",
    plot_name: str = "VGG16_tobacco_plot",
    plot_format: str = "png",
):
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


def save_classification_report(
    classification_report,
    output_dir,
    log_output=True,
    file_name="VGG16_tobacco_report.txt",
):
    if log_output:
        logger.info(f"Classification Report:\n{classification_report}")

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as file:
        file.write(classification_report)
    logger.info(f"Classification report saved as {file_name}")


if __name__ == "__main__":
    path_to_input_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "in"
    )
    data_dir = initialize_data_directory(path_to_input_folder)
    model = model_pipeline(optimizer_type="Adam")

    X, y = load_image_training_data(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    y_train, y_val, y_test = binarize_and_fit_labels(y_train, y_val, y_test)

    data_gen = augment_training_data(X_train, y_train, use_augmentation=True)

    logger.info("Starting model training.")
    H = model.fit(
        data_gen, validation_data=(X_val, y_val), batch_size=128, epochs=10, verbose=1
    )
    logger.info("Model training completed.")

    save_trained_model(model, os.path.join("out", "models"), model_format="keras")

    predictions = model.predict(X_test, batch_size=128)
    labels = get_unique_labels_from_subdirs(data_dir)

    VGG16_report = classification_report(
        y_test.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=labels,
        zero_division=1,
    )

    save_classification_report(
        VGG16_report, "out", file_name="VGG16_tobacco_report.txt"
    )
    plot_history(
        H,
        10,
        save_plot=True,
        output_dir="out",
        plot_name="VGG16_tobacco_plot",
        plot_format="pdf",
    )
