## Capture images manually by keyboard input

import pyrealsense2 as rs
import cv2
import math
import numpy as np
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
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #config depth stream
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        
## Start streaming
profile = pipeline.start(config)  # start the pipeline

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color  #与align to color stream
align = rs.align(align_to)

## Define the filters
decimation_filter = rs.decimation_filter()  
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()


## Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()


def capture_rgbd_images():

    ## Get frameset of color and depth
    frames = pipeline.wait_for_frames()  
    aligned_frames = align.process(frames)    # Align the depth frame to color frame
    color_frame = aligned_frames.get_color_frame()   # get color frame
    aligned_depth_frame = aligned_frames.get_depth_frame()  #get aligned frame, ! aligned_depth_frame is a 640x480 depth image
    
    ## filter the aligned depth frames
    aligned_depth_frames = decimation_filter.process(aligned_depth_frame)
    aligned_depth_frames = spatial_filter.process(aligned_depth_frames)
    aligned_depth_frames = temporal_filter.process(aligned_depth_frames)
    
    ## get the parameter of the camera
    intr = color_frame.profile.as_video_stream_profile().intrinsics   # intrinsic paramter
    
    # get depth intrinsic parameter, used for transformation from pixel coordinates to camera coordinates
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  
    
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': depth_scale
                         }
    
    #######!!!!!!! Get the depth and color image and Convert images to numpy arrays
    depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=float)  # depth image default-16bit）
    color_image = np.asanyarray(color_frame.get_data())  # color image
    
    
    ## Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # convent depth image to 8bit
    depth_colormap = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
    
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape
    
    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images_depth_colormap = np.hstack((resized_color_image, depth_colormap))    #display the color image alongside the color-mapped depth image
        
    else:
        images_depth_colormap = np.hstack((color_image, depth_colormap))
    
    
    ## Align channel of depth image with color image
    # Depth image is 1 channel, color is 3 channels !
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # 3-channel depth image
    depth_image_3d_8bit = np.dstack((depth_image_8bit,depth_image_8bit,depth_image_8bit))  # 3channel 8bit depth image
    
    #return intrinsic_paramter, depth_parameter, color_image, depth_image, color-mapped depth image, 3-channel_depth_image, aligned_depth_frame
    return intr, depth_intrin, color_image, depth_image, images_depth_colormap, depth_image_3d, aligned_depth_frame



if __name__ == "__main__":
    image_num = 0
    
    while True:
    
        intr, depth_intrin, color_image, depth_image, images_depth_colormap, depth_image_3d, aligned_depth_frame = capture_rgbd_images()
        
        image_savepath = "/home/zhanfeng/camera_ws/src/Realsense_python/camera_l515_test/3D_reconstruction_images/"
        color_image_filename = f'color_image_{image_num}.jpg'
        depth_image_filename = f'depth_image_{image_num}.jpg'
        color_depth_image_filename = f'color_depth_image_{image_num}.jpg'
        
        
        #  Show image
        cv2.namedWindow('RGB color image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB color image', color_image)  # display the RBG image
        
        cv2.namedWindow('Depth image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth image', depth_image)    # display the depth image (white-black style)
        
        cv2.namedWindow('Color-mapped depth image', cv2.WINDOW_AUTOSIZE)  #Display the RBG image and the color-mapped depth image (opencv_viewer_example.py)
        cv2.imshow('Color-mapped depth image', images_depth_colormap)    #containing both the color image and depth image above

        key = cv2.waitKey(1)   # the time for image showing in millisecond
        
        
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
        # Press space or 's' to save and write the image
        elif key & 0xFF == ord('s') or key == 32:  
            print(f'Capturing image_{image_num}')
            cv2.imwrite(image_savepath + color_image_filename, color_image)
            cv2.imwrite(image_savepath + depth_image_filename, depth_image)
            cv2.imwrite(image_savepath + color_depth_image_filename, images_depth_colormap)          
            image_num += 1
        
    cv2.destroyAllWindows()




