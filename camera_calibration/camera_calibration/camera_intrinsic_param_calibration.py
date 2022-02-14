# camera intrinsic parameter calibration

import cv2
import numpy as np

image_num = 30

## parameters for Charuco board
params = cv2.aruco.DetectorParameters_create()

square_x = 5                           # Number of squares in X direction
square_y = 7                           # Number of squares in Y direction
square_length = 38                    # Square side length (in millimeters)
marker_length = 30                    # Marker side length (in millimeters)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)    #dictionary

## create board
board = cv2.aruco.CharucoBoard_create(square_x, square_y, square_length, marker_length, dictionary)

image_savepath = "/home/zhanfeng/camera_ws/src/Realsense_python/camera_calibration/Intrinsic_calibration_images/"


all_corners = []
all_ids = []

for i in range(image_num):
    # read image
    image_filename = f'intrinsic_calibration_image_{i}.png'
    image = cv2.imread(image_savepath + image_filename)
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #gray image for markers detection
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)    #color image for display
    
    ## detect inner Aruco corners and ids
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image_gray, dictionary, parameters=params)
    image_color = cv2.aruco.drawDetectedMarkers(image_color, corners, ids)
    
    # calibration
    if len(ids) > 0:
        # charucoCorners, charucoIds
        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, image_gray, board)
        image_color = cv2.aruco.drawDetectedCornersCharuco(image_color, charucoCorners, charucoIds, [0, 0, 255])
        
        if len(charucoCorners)>0:
            all_corners.append(charucoCorners)
            all_ids.append(charucoIds)
            print(i)
    
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape[::-1], None, None)
print("camera intrinsic matrix: \n", cameraMatrix)
print("camera intrinsic distCoeeffs: \n", distCoeffs)

np.savetxt("Calibration_output/camera_intrinsic_matrix.txt", cameraMatrix)
np.savetxt("Calibration_output/camera_intrinsic_distCoeffs.txt", distCoeffs)




