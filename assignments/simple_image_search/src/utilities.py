import random
import pandas as pd
import cv2
import os

def write_dict_to_csv(dictionary, output_path, filename="output.csv"):
    """Write a dictionary to a CSV file."""
    df = pd.DataFrame(list(dictionary.items()), columns=["Filename", "Distance"])
    df.to_csv(os.path.join(output_path, filename) + ".csv", index=False)
    print(f"[SYSTEM] CSV file containing the 5 most similar images has been written to {output_path}")

def compare_histograms(hist_01, hist_02, comp_metric=cv2.HISTCMP_CHISQR):
    """Compare two histograms."""
    return round(cv2.compareHist(hist_01, hist_02, comp_metric), 2)

def load_cv2_image(path):
    """Load an image using cv2."""
    return cv2.imread(path)

def get_highest_value_from_dict(dictionary):
    """Get the highest value from a dictionary."""
    return max(dictionary.values())

def convert_image_to_greyscale(image):
    """Convert an image to greyscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calculate_histogram(image, convert_to_greyscale=False):
    """Calculate the histogram of an image."""
    if convert_to_greyscale:
        grey_scale_image = convert_image_to_greyscale(image)
        hist = cv2.calcHist([grey_scale_image], [0], None, [256], [0, 256])
    else:
        hist = cv2.calcHist([image], [0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

def get_full_paths(directory, files):
    """Return full paths of files in a directory."""
    return [os.path.join(directory, file) for file in files]

def select_random_files(directory, num_to_select=3):
    """Select random files from a directory."""
    files_in_dir = os.listdir(directory)
    return random.sample(files_in_dir, num_to_select)