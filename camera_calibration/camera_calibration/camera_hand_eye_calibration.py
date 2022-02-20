# camera hand eye calibration
# Get the extrinstic matrix between the end-effector and the camera

import numpy as np
import cv2 
#import Function_euler_angle_2_Trans.euler_angle_2_trans as euler_2_Trans
import Function_euler_angle_2_Trans

image_num = 12

## parameters for Charuco board
params = cv2.aruco.DetectorParameters_create()

square_x = 5                           # Number of squares in X direction
square_y = 7                           # Number of squares in Y direction
square_length = 0.038                    # Square side length (in millimeters)
marker_length = 0.030                    # Marker side length (in millimeters)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)    #dictionary

## create board
board = cv2.aruco.CharucoBoard_create(square_x, square_y, square_length, marker_length, dictionary)

image_savepath = "/home/zhanfeng/camera_ws/src/Realsense_python/camera_calibration/Hand_eye_calibration_images_saved/Hand_eye_calibration_images12/"


# read the camera matrix and dist_coeffs
#camera_matrix = np.loadtxt("Calibration_output/camera_intrinsic_matrix.txt", cameraMatrix)
#dist_coeffs = np.loadtxt("Calibration_output/camera_intrinsic_distCoeffs.txt", distCoeffs)

#use uncalibrated camera intrinsics from the intrinsics.json
camera_matrix = np.array([[676.8134765625, 0, 482.47442626953125],
                          [0, 677.247314453125, 276.1454772949219],
                          [0, 0, 1.]])
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


###(1) Pose estimation, calculating Transformation from board to camera
R_board2camera = np.array([])
T_board2camera = np.array([])

count = []

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
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs, rvec=None, tvec=None)
            
            if retval == True:
                dst, jacobian = cv2.Rodrigues(src=rvec)    
                
                tvec = tvec * 1000    # to millimeter
                
                # save pose
                R_board2camera = np.append(R_board2camera, dst)
                T_board2camera = np.append(T_board2camera, tvec)
                
                count.append(i)
                
                image_color = cv2.aruco.drawAxis(image_color, camera_matrix, dist_coeffs, rvec, tvec, 100)
                
                image_cropped = image_color[160:960, 160:800]
                
                #cv2.imshow(f'hand_eye_calibration_image_{i} pose estimation', image_color)
                print(f"Processing hand eye calibration image - {i}")
                cv2.imshow(f'hand_eye_calibration_image_{i} pose estimation', image_cropped)
                
                cv2.waitKey(1000)    #1s
                cv2.imwrite(f'Calibration_output/hand_eye_calibration_pose_estimation/hand_eye_pose_estimation_{i}.png', image_color)
                cv2.imwrite(f'Calibration_output/hand_eye_calibration_pose_estimation/hand_eye_pose_estimation_cropped_{i}.png', image_cropped)
                cv2.destroyAllWindows()
                
    
T_board2camera = T_board2camera.reshape((len(count),3,1))
R_board2camera = R_board2camera.reshape((len(count),3,3))
cv2.destroyAllWindows()
    

###(2) Calculating Transformation from ur5 end effector to ur5 base

R_end2base = np.array([])
T_end2base = np.array([])

ur5_pose_joints_angle = np.array([[-90.0, 179.9, 0.0, 0.45, -0.10, 0.45],[0.0, 163.0, 90.0, 0.34, -0.1, 0.3],[0.0, 150.0, 90.0, 0.30, -0.1, 0.44],[30.0, 152.0, 130.0, 0.32, -0.219, 0.22],[60.0, 158.0, 150.0, 0.355, -0.27, 0.38],[-90.0, -152.0, 30.0, 0.45, -0.26, 0.28],[-60.0, -156.0, 35.0, 0.55, -0.27, 0.34],[-30.0, -145.0, 75.0, 0.6, -0.20, 0.22],[0.0, -157.0, 60.0, 0.60, -0.08, 0.35],[45.0, -154.0, 135.0, 0.58, 0.08, 0.32],[90.0, -152.0, 135.0, 0.45, 0.2, 0.4],[-45.0, 152.0, -45.0, 0.3, 0.08, 0.32]])

ur5_pose_joints_angle = ur5_pose_joints_angle[count] #align number of pose to number of images

for i in range(ur5_pose_joints_angle.shape[0]):
    euler_angle = ur5_pose_joints_angle[i]
    #r_end2base, t_end2base = euler_2_Trans(euler_angle)
    r_end2base, t_end2base = Function_euler_angle_2_Trans.euler_angle_2_trans(euler_angle)
    R_end2base = np.append(R_end2base, r_end2base)
    T_end2base = np.append(T_end2base, t_end2base)

R_end2base = R_end2base.reshape((ur5_pose_joints_angle.shape[0],3,3))
T_end2base = T_end2base.reshape((ur5_pose_joints_angle.shape[0],3,1))


###(3) Hand eye calibration: calculating Transformation from camera to ur5 end effector

R_camera2end, T_camera2end = cv2.calibrateHandEye(R_end2base, T_end2base, R_board2camera, T_board2camera, method=cv2.CALIB_HAND_EYE_HORAUD)  # method??
print('R_camera2end: \n', R_camera2end)
print('T_camera2end: \n', T_camera2end)
    
Trans_camera2end = np.c_[R_camera2end, T_camera2end]
Trans_camera2end = np.r_[Trans_camera2end, np.array([[0,0,0,1]])]
np.savetxt('Calibration_output/hand_eye_calibration_matrix.txt', Trans_camera2end)
print("Hand eye calibration output matrix: \n", Trans_camera2end)




