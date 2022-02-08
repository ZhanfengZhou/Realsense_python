import pyrealsense2 as rs
import numpy as np
import cv2
import time

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
    print("The demo requires Depth camera with Color sensor")
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
    
    
    cv2.imwrite('Color_image_capture.png', color_image)
    
    
    return color_image, image_resized
    
    
    
if __name__ == "__main__":
    while True:
    
        color_image, image_resized = capture_images()

        
        ##  Show image
        cv2.namedWindow('RGB color image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB color image', color_image)  # display the RBG image
        #cv2.imshow('RGB color image2', image_resized)  
        
        key = cv2.waitKey(1)   # the time for image showing in millisecond
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
        
        time.sleep(5)
        
    cv2.destroyAllWindows()


