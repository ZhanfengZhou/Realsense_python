# camera hand eye calibration
# Get the extrinstic matrix between the end-effector and the camera

import numpy as np
import cv2 
#import Function_euler_angle_2_Trans.euler_angle_2_trans as euler_2_Trans
import Function_euler_angle_2_Trans

# read the camera matrix and dist_coeffs
#camera_matrix = np.loadtxt("Calibration_output/camera_intrinsic_matrix.txt", cameraMatrix)
#dist_coeffs = np.loadtxt("Calibration_output/camera_intrinsic_distCoeffs.txt", distCoeffs)

#use uncalibrated camera intrinsics from the intrinsics.json
camera_matrix = np.array([[676.8134765625, 0, 482.47442626953125],
                          [0, 677.247314453125, 276.1454772949219],
                          [0, 0, 1.]])
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

count = []

###(1) Calculating Transformation from ur5 end effector to ur5 base

image_num = 5
ur5_pose_euler_angle = np.array([[-90.0, 179.9, 0.0, 0.45, -0.10, 0.5],[0.0, 150.0, 90.0, 0.22, -0.10, 0.32],[-90.0, -150.0, 0.0, 0.42, -0.35, 0.32],[0.0, -150.0, 60.0, 0.60, -0.10, 0.28],[90.0, -150.0, 90.0, 0.4, 0.15, 0.32]])

R_end2base = np.array([])
T_end2base = np.array([])
for i in range(ur5_pose_euler_angle.shape[0]):
    euler_angle = ur5_pose_euler_angle[i]
    r_end2base, t_end2base = Function_euler_angle_2_Trans.euler_angle_2_trans(euler_angle)
    R_end2base = np.append(R_end2base, r_end2base)
    T_end2base = np.append(T_end2base, t_end2base)

R_end2base = R_end2base.reshape((ur5_pose_euler_angle.shape[0],3,3))
T_end2base = T_end2base.reshape((ur5_pose_euler_angle.shape[0],3,1))
# change to mm


Trans_end2base = []
for i in range(ur5_pose_euler_angle.shape[0]):
    Trans_end2base_i = np.c_[R_end2base[i], T_end2base[i]]
    Trans_end2base_i = np.r_[Trans_end2base_i, np.array([[0,0,0,1]])]
    Trans_end2base.append(Trans_end2base_i)


###(2) Calculating Transformation from camera to ur5 end effector
Trans_camera2end = np.loadtxt("3D_reconstruction_images/hand_eye_calibration_matrix.txt")

Trans_camera2base = []
for i in range(ur5_pose_euler_angle.shape[0]):
    Trans_camera2base_i = np.matmul(Trans_end2base[i], Trans_camera2end)
    Trans_camera2base.append(Trans_camera2base_i)
    Trans_camera2base[i][0][3] /= 1000
    Trans_camera2base[i][1][3] /= 1000
    Trans_camera2base[i][2][3] /= 1000
    np.savetxt(f'3D_reconstruction_images/Matrix_camera2base/camera2base_matrix_for_image_{i}.txt', Trans_camera2base[i])




