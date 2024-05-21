from pathlib import Path
import ssl
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm

from utilities import (
    convert_dict_to_table,
    convert_labels_to_class_name,
    save_classification_report,
    preprocess_image_data
)

# disable SSL certificate verification, not working for virtual environment
ssl._create_default_https_context = ssl._create_unverified_context


def train_and_fit_logistic_regression_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_grid_search: bool = False,
    use_cross_validation: bool = False,
    cross_validation_folds: int = 10,
    parameters: Optional[Dict] = None,
) -> LogisticRegression:
    """
    Trains and fits a logistic regression classifier.

    Parameters:
        X_train (np.ndarray): The input training data.
        y_train (np.ndarray): The target training data.
        use_grid_search (bool, optional): Whether to use grid search to find the best parameters. Defaults to True.
        use_cross_validation (bool, optional): Whether to use cross-validation. Defaults to False.
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
            verbose=True,
            tol=0.1,
            multi_class="multinomial",
        ).fit(X_train, y_train)

        if use_cross_validation:
            # Optionally perform k-fold cross-validation
            print(
                "[INFO] Performing cross-validation with {cross_validation_folds} folds..."
            )
            scores = cross_val_score(
                model, X_train, y_train, cv=cross_validation_folds, verbose=3
            )
            print(f"[INFO] Cross-validation scores: {scores}")
            print(f"[INFO] Average cross-validation score: {scores.mean()}")

    return model


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
        "penalty": ["l2", "none"],
        "solver": ["lbfgs", "liblinear", "saga"],
        "C": [0.001],
        "random_state": [24],
        "max_iter": [500],
        "verbose": [False],
        "tol": [0.1, 0.01, 0.001],
        "multi_class": ["ovr", "multinomial"],
    }

    # train and fit classifier
    classifier = train_and_fit_logistic_regression_classifier(
        X_train_processed,
        y_train_processed,
        use_grid_search=False,
        parameters=grid_search_parameters,
        use_cross_validation=False,
        cross_validation_folds=10,
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
