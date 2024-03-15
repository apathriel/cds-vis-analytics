import os
import sys
import cv2

import numpy as np
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

def load_data():
    pass

def process_data():
    initial_list_test = []
    for image in X_train:
        initial_list_test.append(flatten_array(normalize_pixel_values(convert_image_to_greyscale(image))))    
    print(len(initial_list_test[0]))  
    print(initial_list_test[0])
    print(np.array(initial_list_test))

def convert_image_to_greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize_pixel_values(img):
    return cv2.normalize(img, img, 0, 1.0, cv2.NORM_MINMAX)

def flatten_array(arr):
    return arr.flatten()

process_data()