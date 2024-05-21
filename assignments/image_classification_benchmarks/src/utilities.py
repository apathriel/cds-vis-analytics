from pathlib import Path
from typing import Dict
import cv2
import numpy as np
from tqdm import tqdm

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


def preprocess_image(images_to_process: np.ndarray) -> np.ndarray:
    """
    Preprocesses image data by converting images to grayscale and normalizing pixel values.

    Parameters:
        images_to_process (np.ndarray): Array of images to be processed.

    Returns:
        np.ndarray: Processed image data with normalized pixel values.

    """
    print("[INFO] Processing image data...")
    
    # Initialize an empty list to store the processed images
    processed_images = []

    # Loop over each image in the input array
    for image in tqdm(images_to_process, desc="Processing images"):
        # Convert the image to grayscale
        grayscale_image = convert_image_to_greyscale(image)
        
        # Normalize the pixel values of the image
        scaled_image = grayscale_image / 255.0

        # Flatten the image
        flattened_image = scaled_image.flatten()
        
        # Add the processed image to the list
        processed_images.append(flattened_image)

    print("[INFO] Image data has been processed!")
    
    # Convert the list of processed images to a numpy array and reshape it
    return np.array(processed_images)
