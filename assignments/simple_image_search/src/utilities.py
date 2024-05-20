import logging
from pathlib import Path
import random
import timeit
from typing import Any, Callable, Dict, List, Optional, Union, Generator

import cv2
import numpy as np
import pandas as pd


def get_tqdm_parameters() -> Dict[str, Any]:
    return {
        "desc": "Processing images",
        "leave": True,
        "ncols": 100,
        "unit": " images",
        "unit_scale": True,
    }


def image_generator(
    dataset_path: Path, valid_formats: List[str] = [".jpg", ".png"]
) -> Generator[Path, None, None]:
    """
    Yield images one by one from a dataset.

    Parameters:
    - dataset_path (Path): The path to the dataset directory.
    - valid_formats (List[str]): List of valid file formats.

    Yields:
    - Path: The path to each image in the dataset.
    """
    for image_path in dataset_path.iterdir():
        if image_path.suffix.lower() in valid_formats:
            yield image_path


def write_dict_to_csv(
    dictionary: Dict, output_path: Path, filename: str = "output"
) -> None:
    """
    Write a dictionary to a CSV file.

    Parameters:
        dictionary (Dict): The dictionary to be written to the CSV file.
        output_path (Path): Path to the output directory where CSV file will be saved.
        filename (str, optional): The name of the CSV file (extension is added). Defaults to "output".

    Returns:
        None
    """
    df = pd.DataFrame(list(dictionary.items()), columns=["Filename", "Distance"])

    try:
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Permission denied when trying to create directory: {output_path}")
        return
    except OSError as e:
        print(f"OS error occurred when trying to create directory: {e}")
        return

    df.to_csv(output_path / (filename + ".csv"), index=False)
    print(
        f"[SYSTEM] CSV file containing the 5 most similar images has been written to {output_path}"
    )


def compare_histograms(
    hist_01: np.ndarray, hist_02: np.ndarray, comp_metric: int = cv2.HISTCMP_CHISQR
) -> float:
    """
    Compare two histograms using a specified comparison metric.

    Parameters:
        hist_01 (np.ndarray): The first histogram to compare.
        hist_02 (np.ndarray): The second histogram to compare.
        comp_metric (int): The comparison metric to use (default: cv2.HISTCMP_CHISQR).

    Returns:
        float: The result of the histogram comparison, rounded to 2 decimal places.
    """
    return round(cv2.compareHist(hist_01, hist_02, comp_metric), 2)


def timing_decorator(func: Callable[..., Any], logger: Optional[logging.Logger] = None):
    """
    A decorator that measures the execution time of a function and logs or prints the duration.

    Parameters:
        func (Callable[..., Any]): The function whoose execution time is to be measured.
        logger (Optional[logging.Logger], optional): The logger to be used for logging the duration. Defaults to None.

    Returns:
        Callable[..., Any]: The decorated function.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        if logger:
            logger.debug(f"{func.__name__} took {end - start} seconds to run")
        print(f"{func.__name__} took {end - start} seconds to run")
        return result

    return wrapper


def load_cv2_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image using the cv2 module.

    Parameters:
    image_path (Union[str, Path]): The path to the image file.

    Returns:
    np.ndarray: The loaded image as a NumPy array.
    """
    return cv2.imread(str(image_path))


def get_highest_value_from_dict(dictionary: Dict) -> float:
    """
    Get the highest value from a dictionary.

    Parameters:
    dictionary (Dict): dictionary whose values are to be compared.

    Returns:
    float: The highest value in the given dictionary.
    """
    return max(dictionary.values())


def convert_image_to_greyscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to greyscale.

    Parameters:
        image (numpy.ndarray): The input image to be converted.

    Returns:
        numpy.ndarray: The greyscale version of the input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def calculate_histogram(
    image: np.ndarray, convert_to_greyscale: bool = False
) -> np.ndarray:
    """
    Calculate the histogram of an image.

    Parameters:
    - image: numpy.ndarray
        The input image for which the histogram needs to be calculated.
    - convert_to_greyscale: bool, optional
        Flag indicating whether to convert the image to grayscale before calculating the histogram.
        Default is False.

    Returns:
    - hist: numpy.ndarray
        The calculated histogram of the input image.
    """
    if convert_to_greyscale:
        grey_scale_image = convert_image_to_greyscale(image)
        hist = cv2.calcHist([grey_scale_image], [0], None, [256], [0, 256])
    else:
        hist = cv2.calcHist(
            [image], [0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256]
        )
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)


def get_full_paths(directory: Path, files: List) -> List[Path]:
    """
    Return a list of full paths for files in a directory.

    Parameters:
    directory (Path): The directory path.
    files (List): A list of file names.

    Returns:
    List: A list of full paths for the files in the directory.
    """
    return [directory / file for file in files]


def select_random_files(directory: Path, num_to_select: int = 3) -> List[Path]:
    """
    Pseudo random selection of given num of files from a directory.

    Parameters:
        directory (Path): The directory from which to select random files.
        num_to_select (int, optional): The number of files to select. Defaults to 3.

    Returns:
        List: A list of randomly selected files from the directory.
    """
    files_in_dir = list(directory.iterdir())
    return random.sample(files_in_dir, num_to_select)
