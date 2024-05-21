from pathlib import Path
import ssl
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm

# disable SSL certificate verification, not working for virtual environment
ssl._create_default_https_context = ssl._create_unverified_context


def preprocess_image_data(images_to_process: np.ndarray) -> np.ndarray:
    """
    Preprocesses image data by converting images to grayscale and normalizing pixel values.

    Parameters:
        images_to_process (np.ndarray): Array of images to be processed.

    Returns:
        np.ndarray: Processed image data with normalized pixel values.

    """
    print("[INFO] Processing image data...")
    processed_images = np.array(
        [
            normalize_pixel_values(convert_image_to_greyscale(image))
            for image in tqdm(images_to_process, desc="Processing images")
        ]
    )
    print("[INFO] Image data has been processed!")
    return np.array(processed_images).reshape(-1, 1024)


def convert_image_to_greyscale(img: np.ndarray) -> np.ndarray:
    """
    Converts an image to grayscale using cv2.

    Parameters:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_pixel_values(img: np.ndarray) -> np.ndarray:
    """
    Normalizes the pixel values of an image using v2.

    Parameters:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The normalized image.

    """
    return cv2.normalize(img, img, 0, 1.0, cv2.NORM_MINMAX)


def train_and_fit_logistic_regression_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_grid_search: bool = True,
    parameters: Optional[Dict] = None,
) -> LogisticRegression:
    """
    Trains and fits a logistic regression classifier.

    Parameters:
        X_train (np.ndarray): The input training data.
        y_train (np.ndarray): The target training data.
        use_grid_search (bool, optional): Whether to use grid search to find the best parameters. Defaults to True.
        parameters (Optional[Dict], optional): The parameters for grid search. Defaults to None.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """

    print("[INFO] Training logistic regression classifier...")
    if use_grid_search and parameters is not None:
        print("[INFO] Using grid search to find best parameters...")
        clf = GridSearchCV(
            LogisticRegression(random_state=24, max_iter=1000, verbose=False),
            parameters,
        )
        model = clf.fit(X_train, y_train)
        print(f"Best parameters: {clf.best_params_}")
    else:
        model = LogisticRegression(
            penalty="l2",
            solver="saga",
            C=0.001,
            random_state=24,
            max_iter=1000,
            verbose=False,
            tol=0.001,
            multi_class="auto",
        ).fit(X_train, y_train)
    return model


def convert_dict_to_table(data: Dict) -> np.ndarray:
    """
    Converts a dictionary into a numpy array table.

    Parameters:
        data (Dict): The dictionary to be converted.

    Returns:
        np.ndarray: The numpy array table containing the keys and values from the dictionary.
    """
    return np.column_stack((list(data.keys()), list(data.values())))


def convert_labels_to_class_name(
    data_labels: np.ndarray, label_names: Dict
) -> np.ndarray:
    """
    Converts numeric labels to corresponding class names.

    Parameters:
        data_labels (np.ndarray): Array of numeric labels.
        label_names (Dict): Dictionary mapping label keys to class names.

    Returns:
        np.ndarray: Array of class names corresponding to the input labels.
    """
    label_key_list = list(label_names.values())
    return np.array([label_key_list[label] for label in data_labels])


def save_classification_report(
    classification_report: str,
    output_dir: Path,
    file_name: str = "logistic_classification_report.txt",
) -> None:
    """
    Save the classification report to a file.

    Parameters:
        classification_report (str): The classification report to be saved.
        output_dir (Path): The directory where the file will be saved.
        file_name (str, optional): The name of the file. Defaults to "logistic_classification_report.txt".
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / file_name
    with open(file_path, "w") as file:
        file.write(classification_report)
    print(f"[INFO] Classification report saved as {file_name}")


def main():
    # load cifar10 dataset with tensorflow
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # class names for cifar10 dataset
    cifar10_class_names = {
        "0": "airplane",
        "1": "automobile",
        "2": "bird",
        "3": "cat",
        "4": "deer",
        "5": "dog",
        "6": "frog",
        "7": "horse",
        "8": "ship",
        "9": "truck",
    }
    cifar10_label_conversion_table = convert_dict_to_table(cifar10_class_names)

    # specify output directory path, use file as base, rather th
    output_dir_path = Path(__file__).parent / ".." / "out" / "logistic_regression"

    # preprocess image data
    X_train_processed = preprocess_image_data(X_train)
    X_test_processed = preprocess_image_data(X_test)
    y_train_processed = convert_labels_to_class_name(
        np.ravel(y_train), cifar10_class_names
    )
    y_test_processed = convert_labels_to_class_name(
        np.ravel(y_test), cifar10_class_names
    )

    grid_search_parameters = {
    'penalty': ['l2', 'none'],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'C': [0.001],
    'random_state': [24],
    'max_iter': [500],
    'verbose': [False],
    'tol': [0.1, 0.01, 0.001],
    'multi_class': ['auto', 'ovr', 'multinomial'],
}

    # train and fit classifier
    classifier = train_and_fit_logistic_regression_classifier(
        X_train_processed,
        y_train_processed,
        use_grid_search=False,
        parameters=grid_search_parameters,
    )

    # predict test data
    y_pred = classifier.predict(X_test_processed)

    save_classification_report(
        classification_report(
            y_test_processed, y_pred, target_names=cifar10_label_conversion_table[:, 1]
        ),
        output_dir_path,
    )


if __name__ == "__main__":
    main()
