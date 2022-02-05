# Reference: https://blog.csdn.net/zhy29563/article/details/119039163

# Create a ChArUco board image for camera calibration

import cv2 as cv
import numpy as np

# 创建ChArUco标定板
dictionary = cv.aruco.getPredefinedDictionary(dict=cv.aruco.DICT_6X6_250)
board = cv.aruco.CharucoBoard_create(squaresY=7,
                                     squaresX=5,
                                     squareLength=0.04,
                                     markerLength=0.02,
                                     dictionary=dictionary)
img_board = board.draw(outSize=(600, 500), marginSize=10, borderBits=1)
cv.imwrite(filename='charuco.png', img=img_board, params=None)


