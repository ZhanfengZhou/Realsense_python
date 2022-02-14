## Capture images manually by keyboard input

import pyrealsense2 as rs
import cv2

import math
import numpy as np


## Configure depth and color streams
pipeline = rs.pipeline()  #Create a realsenes pipeline
        
config = rs.config()   # Create a config
        
## Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
#for our camera L515, the device_product_line == 'L500'
        
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
   raise Exception('No RGB camera found!')
   exit(0)
        
## define the sensor parameter for close range image process
sensor = pipeline_profile.get_device().query_sensors()[0]

sensor.set_option(rs.option.laser_power, 100)
sensor.set_option(rs.option.confidence_threshold, 1)
sensor.set_option(rs.option.min_distance, 0)
sensor.set_option(rs.option.enable_max_usable_range, 0)
sensor.set_option(rs.option.receiver_gain, 18)
sensor.set_option(rs.option.post_processing_sharpening, 3)
sensor.set_option(rs.option.pre_processing_sharpening, 5)
sensor.set_option(rs.option.noise_filtering, 6)


## Configure the pipeline to stream different resolutions of color and depth streams
        
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        
## Start streaming
profile = pipeline.start(config)  # start the pipeline


def capture_images():

    ## Get frameset of color
    frames = pipeline.wait_for_frames()  
    color_frame = frames.get_color_frame()   # get color frame
    
    
    ## Get the depth and color image and Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())  # color image
    
    image_resized = cv2.resize(color_image,(int(1280/2),int(720/2)))

    return color_image, image_resized


if __name__ == "__main__":
    image_num = 0
    
    while True:
    
        color_image, image_resized = capture_images()
        
        image_savepath = "/home/zhanfeng/camera_ws/src/Realsense_python/camera_calibration/Intrinsic_calibration_images/"
        image_filename = f'intrinsic_calibration_image_{image_num}.png'
        
        ##  Show image
        cv2.namedWindow('intrinsic calibration RGB image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('intrinsic calibration RGB image', color_image)  # display the RBG images

        key = cv2.waitKey(1)   # the time for image showing in millisecond
        
        
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
        # Press space or 's' to save and write the image
        elif key & 0xFF == ord('s') or key == 32:  
            print(f'Capturing hand_eye_calibration_image_{image_num}.png')
            cv2.imwrite(image_savepath + image_filename, color_image)
            image_num += 1
        
    cv2.destroyAllWindows()












