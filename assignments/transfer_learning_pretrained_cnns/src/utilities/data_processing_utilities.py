from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import (
    img_to_array,
    load_img,
)
from tqdm import tqdm

from .logging_utilities import get_logger

logger = get_logger(__name__)


# Data processing, loading and prep
def initialize_data_directory(data_path):

    # Check if the directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"The directory {data_path} does not exist.")

    # Check if the directory is accessible
    if not data_path.is_dir():
        raise NotADirectoryError(f"The path {data_path} is not a directory.")

    return data_path


def create_label_list_from_subdirs(image_files):
    labels = []
    for image_file in image_files:
        labels.append(Path(image_file).parent.name)
    labels = np.array(labels)
    return labels


def get_unique_labels_from_subdirs(directory):
    return [item.name for item in Path(directory).iterdir() if item.is_dir()]


def load_and_preprocess_training_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        logger.info(f"Loading images...")
        # load all jpg files in the data directory, create labels
        image_files = sorted([str(file) for file in Path(data_path).rglob("*.jpg")])
        labels = create_label_list_from_subdirs(image_files)
        # load images using keras/pillow
        loaded_images = [
            load_img(img, color_mode="rgb", target_size=(224, 224))
            for img in tqdm(image_files, desc="Loading images...")
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


def split_data(X, y, test_size: float = 0.20, validation_size: float = None, stratify=None):
    if validation_size is None:
        # Only split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify)
        return X_train, X_test, y_train, y_test
    else:
        # First split to separate out the training set
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + validation_size, stratify=stratify
        )

        # Second split to separate out the validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size / (test_size + validation_size), stratify=y_temp if stratify is not None else None
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


def binarize_and_fit_labels(y_train, y_test, val_split=None):
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    if val_split is not None:
        y_train, y_val = train_test_split(y_train, test_size=val_split, stratify=y_train)

        return y_train, y_val, y_test

    return y_train, y_test
