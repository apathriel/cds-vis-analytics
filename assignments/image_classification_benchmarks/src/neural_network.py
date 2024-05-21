import cv2
from pathlib import Path
import ssl
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm

# disable SSL certificate verification, not working for virtual environment
ssl._create_default_https_context = ssl._create_unverified_context


def preprocess_image_data(images_to_process: np.ndarray) -> np.ndarray:
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_pixel_values(img: np.ndarray) -> np.ndarray:
    return cv2.normalize(img, img, 0, 1.0, cv2.NORM_MINMAX)


def train_and_fit_neural_network_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_activation: str = "ReLU",
    use_grid_search: bool = False,
    parameters: Optional[Dict] = None,
) -> MLPClassifier:
    print("[INFO] Training neural network classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=100,
        random_state=42,
        activation=classifier_activation,
        early_stopping=True,
        verbose=False,
    )
    if use_grid_search and parameters is not None:
        print("[INFO] Using grid search to find best parameters...")
        clf = GridSearchCV(clf, parameters, verbose=2)
        model = clf.fit(X_train, y_train)
        print(f"Best parameters: {clf.best_params_}")
    else:
        model = clf.fit(X_train, y_train)
    return model


def convert_dict_to_table(data: Dict) -> np.ndarray:
    return np.column_stack((list(data.keys()), list(data.values())))


def convert_labels_to_class_name(
    data_labels: np.ndarray, label_names: Dict
) -> np.ndarray:
    label_key_list = list(label_names.values())
    return np.array([label_key_list[label] for label in data_labels])


def save_classification_report(
    classification_report: str,
    output_dir: Path,
    file_name: str = "neural_classification_report.txt",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / file_name
    with open(file_path, "w") as file:
        file.write(classification_report)
    print(f"[INFO] Classification report saved as {file_name}")


def plot_loss_curve(
    classifier: MLPClassifier,
    output_dir: str,
    file_name: str = "neural_loss_curve",
    file_format: str = "png",
    save_file: bool = True,
) -> None:
    plt.plot(classifier.loss_curve_)
    plt.title("Neural Network Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss score")

    if save_file:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / (file_name + "." + file_format))
        print(f"[INFO] Loss curve saved as {file_name}")


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
    output_dir_path = Path(__file__).parent / ".." / "out" / "neural_network"

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
        "hidden_layer_sizes": [(128,), (128, 64), (256, 128)],
        "activation": ["tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.05],
        "learning_rate": ["constant", "adaptive"],
        "tol": [0.0001, 0.001, 0.01, 0.1],
    }

    # train and fit classifier
    classifier = train_and_fit_neural_network_classifier(
        X_train_processed,
        y_train_processed,
        use_grid_search=True,
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
    plot_loss_curve(classifier, output_dir_path)


if __name__ == "__main__":
    main()
