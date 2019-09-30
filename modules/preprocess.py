import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 60), interpolation = cv2.INTER_AREA)
    gray = gray[15:, :] / 255
    return gray
