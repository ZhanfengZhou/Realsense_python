import pyrealsense2 as rs
import numpy as np
import cv2
import json
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

#config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 1024, 768, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #config depth stream
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
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
print("Depth Scale is: " , depth_scale)



def get_aligned_images():
    
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
    # save intrinsic parameter
    with open('./intrinsics.json', 'w') as fp:
        json.dump(camera_parameters, fp)
    
    
    ## Get the depth and color image and Convert images to numpy arrays
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
    
    ## Remove background - Set pixels further than clipping_distance to grey
    # We will be removing the background of objects more than 'clipping_distance_in_meters' meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    grey_color = 153
    color_image_bk_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
    
    
    #return intrinsic_paramter, depth_parameter, color_image, depth_image, color-mapped depth image, 3-channel_depth_image, aligned_depth_frame
    return intr, depth_intrin, color_image, depth_image, images_depth_colormap, depth_image_3d, aligned_depth_frame, color_image_bk_removed
    
if __name__ == "__main__":
    while True:
    
        ## get the aligned image and camera intrinsic param
        intr, depth_intrin, color_image, depth_image, images_depth_colormap, depth_image_3d, aligned_depth_frame, color_image_bk_removed = get_aligned_images()
        
        ## chose the x, y coordinates (now the center point of camera)
        x = 320  
        y = 240
        
        ## get the real z distance of (x, y) point
        dis = aligned_depth_frame.get_distance(x, y)  
        
        
        ## !!!!!! the origin of the camera coordinates is at the center point on the back surface of the camera  
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)  
        # get the real (x, y, z) value in the camera coordinates, which is a 3D array: camera_coordinate
        # camera_coordinate[2] is still the 'dis'，camera_coordinate[0] and camera_coordinate[1] are the real x, y value。
        print(camera_coordinate)
        
        
        ###  Show image
        #cv2.namedWindow('RGB color image', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RGB color image', color_image)  # display the RBG image
        
        #cv2.namedWindow('Depth image', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('Depth image', depth_image)    # display the depth image (white-black style)
        
        cv2.namedWindow('Color-mapped depth image', cv2.WINDOW_AUTOSIZE)  #Display the RBG image and the color-mapped depth image (opencv_viewer_example.py)
        cv2.imshow('Color-mapped depth image', images_depth_colormap)    #containing both the color image and depth image above
        
        cv2.namedWindow('Background-removed color image', cv2.WINDOW_NORMAL)
        cv2.imshow('Background-removed color image', color_image_bk_removed)    # display the background-removed color image (align-depth2color.py)
        
        
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
        
        time.sleep(0.2)   # get data every 0.2 second - 5 Hz.
        
    cv2.destroyAllWindows()


