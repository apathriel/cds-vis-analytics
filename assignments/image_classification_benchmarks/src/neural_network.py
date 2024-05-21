from pathlib import Path
import ssl
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import cifar10

from utilities import (
    convert_dict_to_table,
    convert_labels_to_class_name,
    preprocess_image,
    save_classification_report,
)

# disable SSL certificate verification, not working for virtual environment
ssl._create_default_https_context = ssl._create_unverified_context


def train_and_fit_neural_network_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_parameters: Dict,
    use_grid_search: bool = False,
    grid_search_parameters: Optional[Dict] = None,
    use_cross_validation: bool = False,
    cross_validation_folds: int = 10,
) -> MLPClassifier:
    """
    Trains and fits a neural network classifier.

    Parameters:
        X_train (np.ndarray): The input training data.
        y_train (np.ndarray): The target training data.
        classifier_parameters (Dict): The parameters for the classifier.
        use_grid_search (bool, optional): Whether to use grid search to find the best parameters. Defaults to False.
        grid_search_parameters (Optional[Dict], optional): The parameters for grid search. Defaults to None.
        use_cross_validation (bool, optional): Whether to use cross-validation. Defaults to False.
        cross_validation_folds (int, optional): The number of folds for cross-validation. Defaults to 10.

    Returns:
        MLPClassifier: The trained neural network model.
    """
    print("[INFO] Training neural network classifier...")

    clf = MLPClassifier(**classifier_parameters)

    if use_grid_search and grid_search_parameters is not None:
        print("[INFO] Using grid search to find best parameters...")
        clf = GridSearchCV(clf, grid_search_parameters, verbose=2)
        model = clf.fit(X_train, y_train)
        print(f"Best parameters: {clf.best_params_}")
    else:
        model = clf.fit(X_train, y_train)

    if use_cross_validation:
        print(f"[INFO] Performing cross-validation with {cross_validation_folds} folds...")
        scores = cross_val_score(clf, X_train, y_train, cv=cross_validation_folds)
        print(f"[INFO] Cross-validation scores: {scores}")
        print(f"[INFO] Average cross-validation score: {scores.mean()}")

    return model


def plot_loss_curve(
    classifier: MLPClassifier,
    output_dir: Path,
    file_name: str = "neural_loss_curve",
    file_format: str = "png",
    save_file: bool = True,
) -> None:
    """
    Plots the loss curve of a neural network classifier and saves it as an image file.

    Parameters:
        classifier (MLPClassifier): The trained neural network classifier.
        output_dir (str): The directory where the loss curve image will be saved.
        file_name (str, optional): The name of the loss curve image file. Defaults to "neural_loss_curve".
        file_format (str, optional): The format of the loss curve image file. Defaults to "png".
        save_file (bool, optional): Whether to save the loss curve image file. Defaults to True.
    """
    plt.plot(classifier.loss_curve_)
    plt.title("Neural Network Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss score")

    if save_file:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / (file_name + "." + file_format))
        print(f"[INFO] Loss curve saved as {file_name}")


def main():
    # Starting default classifier parameters
    base_classifier_parameters = {
        "hidden_layer_sizes": (128,),
        "max_iter": 1000,
        "random_state": 42,
        "early_stopping": True,
        "verbose": True,
    }

    # GridSearchCV parameters
    optimized_classifier_parameters = {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.05,
        "max_iter": 500,
        "random_state": 42,
        "verbose": True,
        "early_stopping": True,
        "learning_rate_init": 0.001,
    }

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

    # Preprocess image data
    X_train_processed = preprocess_image(X_train)
    X_test_processed = preprocess_image(X_test)

    y_train_processed = convert_labels_to_class_name(
        np.ravel(y_train), cifar10_class_names
    )
    y_test_processed = convert_labels_to_class_name(
        np.ravel(y_test), cifar10_class_names
    )

    # train and fit classifier
    classifier = train_and_fit_neural_network_classifier(
        X_train_processed,
        y_train_processed,
        classifier_parameters=optimized_classifier_parameters,
        use_grid_search=False,
        grid_search_parameters=None,
    )

    # Perform predictions on test data
    y_pred = classifier.predict(X_test_processed)

    # Save classification report and plot loss curve
    final_classification_report = classification_report(
        y_test_processed, y_pred, target_names=cifar10_label_conversion_table[:, 1]
    )

    save_classification_report(
        classification_report=final_classification_report,
        output_dir=output_dir_path,
        file_name="neural_network_classification_report.txt",
    )
    plot_loss_curve(classifier, output_dir_path)


if __name__ == "__main__":
    main()
