# camera hand eye calibration
# Get the extrinstic matrix between the end-effector and the camera

import numpy as np
import cv2 
import Function_euler_angle_2_Trans.euler_angle_2_trans as euler_2_Trans

image_num = 37

## parameters for Charuco board
params = cv2.aruco.DetectorParameters_create()

square_x = 5                           # Number of squares in X direction
square_y = 7                           # Number of squares in Y direction
square_length = 38                    # Square side length (in millimeters)
marker_length = 30                    # Marker side length (in millimeters)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)    #dictionary

## create board
board = cv2.aruco.CharucoBoard_create(square_x, square_y, square_length, marker_length, dictionary)

image_savepath = "/home/zhanfeng/camera_ws/src/Realsense_python/camera_calibration/Hand_eye_calibration_images/"


# read the camera matrix and dist_coeffs
camera_matrix = np.loadtxt("Calibration_output/camera_intrinsic_matrix.txt", cameraMatrix)
dist_coeffs = np.loadtxt("Calibration_output/camera_intrinsic_distCoeffs.txt", distCoeffs)

R_target2cam = np.array([])
t_target2cam = np.array([])

count = []

### Pose estimation
for i in range(image_num):
    image_filename = f'hand_eye_calibration_image_{i}.png'
    image = cv2.imread(image_savepath + image_filename)
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #gray image for markers detection
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)    #color image for display
    
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image_gray, dictionary, parameters=params)
    image_color = cv2.aruco.drawDetectedMarkers(image_color, corners, ids)
    
    if len(ids) > 0:
        # charucoCorners, charucoIds
        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, image_gray, board)
        image_color = cv2.aruco.drawDetectedCornersCharuco(image_color, charucoCorners, charucoIds, [0, 0, 255])
        
        if len(charucoCorners)>0:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, chboard, mtx, dist, rvec=None, tvec=None)
            
            if retval == True:
                dst, jacobian = cv2.Rodrigues(src=rvec)    
                
                tvec = tvec * 1000    # to millimeter
                
                # save pose
                R_target2cam = np.append(R_target2cam, dst)
                t_target2cam = np.append(t_target2cam, tvec)
                
                count.append(i)
                
                image_color = cv2.aruco.drawAxis(result, mtx, dist, rvec, tvec, 100)
                
                cv2.imshow(f'hand_eye_calibration_image_{i} pose estimation', image_color)
                cv2.waitKey(1000)    #1s
                cv2.destroyAllWindows()
                
    
t_target2cam = t_target2cam.reshape((len(count),3,1))
R_target2cam = R_target2cam.reshape((len(count),3,3))
cv2.destroyAllWindows()
    

### hand eye calibration


Trans_R_end2base = np.array([])
Trans_T_end2base = np.array([])

euler_2_Trans(euler_angle)




















    
