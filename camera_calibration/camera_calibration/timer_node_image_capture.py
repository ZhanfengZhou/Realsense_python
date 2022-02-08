## Capture a image at the 5th second of every 6 seconds

import rclpy
from rclpy.node import Node

import pyrealsense2 as rs
import cv2
import time

import math
import numpy as np


class TimerNodeImageCapture(Node):
    def __init__(self):
        super().__init__("timer_node_image_capture")
        
        # Declare all parameters
        self.declare_parameter("wait_sec_between_image_capture", 6)
        self.declare_parameter("image_capture_time", 5)     #capture a image at the 5th second of every 6 seconds
        self.declare_parameter("image_capture_number", 35)

        # Read parameters
        wait_sec_between_image_capture = self.get_parameter("wait_sec_between_image_capture").value
        self.image_capture_time = self.get_parameter("image_capture_time").value
        self.image_capture_number = self.get_parameter("image_capture_number").value
        
        self.image_savepath = "/home/zhanfeng/camera_ws/src/Realsense_python/camera_calibration/Hand_eye_calibration_images/"

        self.get_logger().info(
            'Capturing {} images for hand-eye calibration at the {}th second of every {} s'.format(
                self.image_capture_number, self.image_capture_time, wait_sec_between_image_capture
            )
        )
        
        self.pipeline = self.init_camera_l515()
        
        #self.publisher_ = self.create_publisher(JointTrajectory, publish_topic, 1)  # not define a publisher yet
        
        self.timer = self.create_timer(wait_sec_between_image_capture, self.timer_callback)
        self.i = 0
    
    def init_camera_l515(self):
            
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
        
        return pipeline

        
    def timer_callback(self):

        ## capturing image
        time.sleep(5)
        
        if self.i < self.image_capture_number:
            self.capture_images()
        elif self.i == 35:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            self.get_logger().info('Capturing {} images finished'.format(self.image_capture_number))
        else:
            self.get_logger().info('timer exceed {} image capture number limits'.format(self.image_capture_number))
        
        self.i += 1
        
    def capture_images(self):
    
        ## Get frameset of color
        frames = self.pipeline.wait_for_frames()  
        color_frame = frames.get_color_frame()   # get color frame
        
        
        ## Get the depth and color image and Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())  # color image
        
        image_resized = cv2.resize(color_image,(int(1280/2),int(720/2)))
        
        image_filename = 'hand_eye_calibration_image_{}.png'.format(self.i)
        cv2.imwrite(self.image_savepath + image_filename, color_image)
        
        self.get_logger().info('Capturing hand_eye_calibration_image_{}.png'.format(self.i))
        
        ##  Show image
        cv2.imshow('Capturing hand_eye_calibration_image_{}'.format(self.i), color_image)
        k = cv2.waitKey(2000)
        
        cv2.destroyAllWindows()
        
        return
    



def main(args=None):
    rclpy.init(args=args)

    timer_node_image_capture = TimerNodeImageCapture()

    rclpy.spin(timer_node_image_capture)
    timer_node_image_capture.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    
    
    
