import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time 

pipeline = rs.pipeline()  #define realsenes pipeline
config = rs.config()   #define realsense config

## define the sensor parameter for close range image process
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

sensor = pipeline_profile.get_device().query_sensors()[0]

sensor.set_option(rs.option.laser_power, 100)
sensor.set_option(rs.option.confidence_threshold, 1)
sensor.set_option(rs.option.min_distance, 0)
sensor.set_option(rs.option.enable_max_usable_range, 0)
sensor.set_option(rs.option.receiver_gain, 18)
sensor.set_option(rs.option.post_processing_sharpening, 3)
sensor.set_option(rs.option.pre_processing_sharpening, 5)
sensor.set_option(rs.option.noise_filtering, 6)

## config and start streamimg
#config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 1024, 768, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #config depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)   #config color stream

profile = pipeline.start(config)  # start the pipeline
align_to = rs.stream.color  #与align to color stream
align = rs.align(align_to)

decimation_filter = rs.decimation_filter()  #define the filters
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()


def get_aligned_images():
    
    ## get frames of color and depth
    frames = pipeline.wait_for_frames()  
    aligned_frames = align.process(frames)  
    color_frame = aligned_frames.get_color_frame()   # get color frame
    aligned_depth_frame = aligned_frames.get_depth_frame()  #get the depth frame
    
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
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }
    # save intrinsic parameter
    with open('./intrinsics.json', 'w') as fp:
        json.dump(camera_parameters, fp)
    
    ## get the depth image with different resolution (8 and 16 bit)
    depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=float)  # depth image default-16bit）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # depth image 8bit
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # 3-channel depth image
    depth_image_3d_8bit = np.dstack((depth_image_8bit,depth_image_8bit,depth_image_8bit))  # 3channel 8bit depth image
    color_image = np.asanyarray(color_frame.get_data())  # color image
    
    #return intrinsic_paramter, depth_parameter, color_image, depth_image、3-channel_depth_image, aligned_depth_frame
    return intr, depth_intrin, color_image, depth_image, depth_image_3d, aligned_depth_frame
    
if __name__ == "__main__":
    while True:
    
        ## get the aligned image and camera intrinsic param
        intr, depth_intrin, color_image, depth_image, depth_image_3d, aligned_depth_frame = get_aligned_images()
        
        ## chose the x, y coordinates (now the center point of camera)
        x = 512
        y = 384
        #x = 320  
        #y = 240
        
        ## get the real z distance of (x, y) point
        dis = aligned_depth_frame.get_distance(x, y)  
        
        ## !!!!!! the origin of the camera coordinates is at the center point on the back surface of the camera  
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)  
        # get the real (x, y, z) value in the camera coordinates, which is a 3D array: camera_coordinate
        # camera_coordinate[2] is still the 'dis'，camera_coordinate[0] and camera_coordinate[1] are the real x, y value。
        print(camera_coordinate)
        
        ## show image
        cv2.namedWindow('Color image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB image',color_image)  #display the RBG image with the openCV2 image show
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
        
        time.sleep(0.2)   # get data every 0.5 second.
        
    cv2.destroyAllWindows()


