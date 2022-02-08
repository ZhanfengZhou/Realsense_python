### Create a ChArUco board image for camera calibration
# reference; https://blog.csdn.net/zhy29563/article/details/119039163

import cv2
import numpy as np

#use uncalibrated camera intrinsics from the intrinsics.json
camera_matrix = np.array([[676.8134765625, 0, 482.47442626953125],
                          [0, 677.247314453125, 276.1454772949219],
                          [0, 0, 1.]])
dist_coefs = np.array([0, 0, 0, 0, 0])


## parameters for Charuco board
params = cv2.aruco.DetectorParameters_create()

square_x = 5                           # Number of squares in X direction
square_y = 7                           # Number of squares in Y direction
square_length = 30                    # Square side length (millimeters)
marker_length = 23                    # Marker side length (milimeters)
margins = 30

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)    #dictionary

## create board
board = cv2.aruco.CharucoBoard_create(square_x, square_y, square_length, marker_length, dictionary)

image_filename = f'/home/zhanfeng/camera_ws/Realsense_python/camera_calibration/Color_image_capture.png'
image = cv2.imread(image_filename)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #gray image for markers detection
image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)    #color image for display

## detect inner Aruco corners and ids
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image_gray, dictionary, parameters=params)

#print('Aruco corners: ', corners)
#print('Aruco corners IDs: ', ids)

if len(ids) > 0:
    print((int(corners[0][0][0][0]), int(corners[0][0][0][1])))
    print((int(corners[1][0][0][0]), int(corners[1][0][0][1])))
    cv2.circle(image_color, (int(corners[0][0][0][0]), int(corners[0][0][0][1])), 8, [0, 255, 0])
    cv2.circle(image_color, (int(corners[1][0][0][0]), int(corners[1][0][0][1])), 8, [0, 255, 0])
    
    # display the color image with aruco corners with ids
    cv2.aruco.drawDetectedMarkers(image_color, corners, ids)
    cv2.namedWindow('image with aruco corner and IDs detected', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image with aruco corner and IDs detected", image_color)
    cv2.waitKey(0)

    ## detect outer Chessboard corners and ids
    retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, image_gray, board)

    if len(charucoIds) > 0:
    
        # display the color image with outer Chessboard  corners with ids
        cv2.aruco.drawDetectedCornersCharuco(image_color, charucoCorners, charucoIds, [0, 0, 255])
        cv2.imwrite('charucoboard_with_corner_detected.png', image_color)
        cv2.namedWindow('image with chessboard corner and IDs detected', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("image with chessboard corner and IDs detected", image_color)
        cv2.waitKey(0)



