import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess_image(img):
    #gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 60), interpolation = cv2.INTER_AREA)
    gray = gray[15:, :] / 255
    #cv2.imshow("Display window", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return gray
