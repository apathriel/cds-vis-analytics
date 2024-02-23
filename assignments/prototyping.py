# need to have all the functionality before implementing Tkinter GUI through branch.
# check if higher than existing values - performance

import os

import sys
sys.path.append(os.path.join(".."))

import cv2

import numpy as np

from utils.imutils import jimshow as show
from utils.imutils import jimshow_channel as show_channel

import matplotlib.pyplot as plt

particular_flower = os.path.join("..", "..", "..", "cds-vis-data", "flowers", "image_0001.jpg")

image = cv2.imread(particular_flower)

