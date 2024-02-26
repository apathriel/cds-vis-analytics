# need to have all the functionality before implementing Tkinter GUI through branch.
# check if higher than existing values - performance
# sys.path.append(os.path.join(".."))

import os
import sys
import cv2
import numpy as np
from utils.imutils import jimshow as show
from utils.imutils import jimshow_channel as show_channel
import matplotlib.pyplot as plt

particular_flower_path = os.path.join("..", "..", "..", "cds-vis-data", "flowers", "image_0001.jpg")
particular_flower_path2 = os.path.join("..", "..", "..", "cds-vis-data", "flowers", "image_0002.jpg")

image = cv2.imread(particular_flower_path)
image2 = cv2.imread(particular_flower_path2)

hist_01 = cv2.calcHist([image],[0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])
hist_02 = cv2.calcHist([image2],[0, 1, 2], None, [255, 255, 255], [0, 256, 0, 256, 0, 256])

normalized_hist_01 = cv2.normalize(hist_01, hist_01, 0, 1.0, cv2.NORM_MINMAX)
normalized_hist_02 = cv2.normalize(hist_02, hist_02, 0, 1.0, cv2.NORM_MINMAX)

comparison = round(cv2.compareHist(normalized_hist_01, normalized_hist_02, cv2.HISTCMP_CHISQR), 2)
