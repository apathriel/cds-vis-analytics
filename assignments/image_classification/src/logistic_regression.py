import os
import ssl

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import cifar10

# disable SSL certificate verification, not working for virtual environment
ssl._create_default_https_context = ssl._create_unverified_context

def preprocess_image_data(images_to_process: np.ndarray) -> np.ndarray:
    print("[INFO] Processing image data...")
    processed_images = np.array([normalize_pixel_values(convert_image_to_greyscale(image)) for image in images_to_process])
    print("[INFO] Image data has been processed!")
    return np.array(processed_images).reshape(-1, 1024)

def convert_image_to_greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize_pixel_values(img):
    return cv2.normalize(img, img, 0, 1.0, cv2.NORM_MINMAX)

def train_and_fit_classifier(X_train, y_train):
    print("[INFO] Training logistic regression classifier...")
    return LogisticRegression(random_state=24, max_iter=1000).fit(X_train, y_train)

def convert_dict_to_table(data: dict) -> np.ndarray:
    return np.column_stack((list(data.keys()), list(data.values())))

def convert_labels_to_class_name(data_labels: np.ndarray, label_names: dict) -> np.ndarray:
    label_key_list = list(label_names.values())
    return np.array([label_key_list[label] for label in data_labels])

def save_classification_report(classification_report, output_dir, file_name="logistic_classification_report.txt"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as file:
        file.write(classification_report)
    print(f"[INFO] Classification report saved as {file_name}")

def main():
    # load cifar10 dataset with tensorflow
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # class names for cifar10 dataset
    cifar10_class_names = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', '4': 'deer', '5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship', '9': 'truck'}
    cifar10_label_conversion_table = convert_dict_to_table(cifar10_class_names)

    # specify output directory path, use file as base, rather th
    output_dir_path = os.path.join(os.path.dirname(__file__), "..", "out", "logistic_regression")

    # preprocess image data
    X_train_processed = preprocess_image_data(X_train)
    X_test_processed = preprocess_image_data(X_test)
    y_train_processed = convert_labels_to_class_name(np.ravel(y_train), cifar10_class_names)
    y_test_processed = convert_labels_to_class_name(np.ravel(y_test), cifar10_class_names)

    # train and fit classifier
    classifier = train_and_fit_classifier(X_train_processed, y_train_processed)

    # predict test data
    y_pred = classifier.predict(X_test_processed)

    save_classification_report(classification_report(y_test_processed, y_pred, target_names=cifar10_label_conversion_table[:, 1]), output_dir_path)

if __name__ == "__main__":
    main()