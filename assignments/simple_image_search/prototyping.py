# need to have all the functionality before implementing Tkinter GUI through branch.
# check if higher than existing values - performance

import os
import sys
import numpy as np
import pandas as pd
import timeit

sys.path.append(os.path.join("..", ".."))

import cv2
from utils.imutils import jimshow as show
from utils.imutils import jimshow_channel as show_channel
import matplotlib.pyplot as plt

def convert_image_to_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compare_histograms(hist_01, hist_02):
    return round(cv2.compareHist(hist_01, hist_02, cv2.HISTCMP_CHISQR), 2)

def calculate_histogram(image, convert_to_greyscale=False):
    if convert_to_greyscale:
        grey_scale_image = convert_image_to_greyscale(image)
        hist = cv2.calcHist([grey_scale_image],[0], None, [256], [0, 256])
    else:
        hist = cv2.calcHist([image],[0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

def load_cv2_image(path):
    return cv2.imread(path)

def get_highest_value_from_dict(dictionary):
    return max(dictionary.values())

def write_dict_to_csv(dictionary, output_path, filename="output.csv"):
    df = pd.DataFrame(list(dictionary.items()), columns=["Filename", "Distance"])
    df.to_csv(os.path.join(output_path, filename) + ".csv", index=False)

def compare_images_in_dataset(dataset_path, target_image, output_path, convert_to_greyscale=False):
    target_image_hist = calculate_histogram(load_cv2_image(os.path.join(dataset_path, target_image)))
    target_image_filename = target_image.split(".")[0] + "_most_similar_images"
    most_similiar_images_5 = {}
    for image in os.listdir(dataset_path):
        if image == target_image:
            continue

        image_to_be_compared = calculate_histogram(load_cv2_image(os.path.join(dataset_path, image)))
        comparison_val = compare_histograms(target_image_hist, image_to_be_compared)

        if len(most_similiar_images_5) < 5:
            most_similiar_images_5.update({image: comparison_val})
        elif len(most_similiar_images_5) == 5 and comparison_val < get_highest_value_from_dict(most_similiar_images_5):
            most_similiar_images_5.pop(max(most_similiar_images_5, key=most_similiar_images_5.get))
            most_similiar_images_5.update({image: comparison_val})
            
    write_dict_to_csv(most_similiar_images_5, output_path, target_image_filename)

""" if __name__ == "__main__": """
flower_dataset_path = os.path.join("data", "input", "flowers")
csv_output_path = os.path.join("data", "output")
target_flower = "image_0001.jpg"

compare_images_in_dataset(flower_dataset_path, target_flower, csv_output_path)