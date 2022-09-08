import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import cv2.aruco as aruco
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

from std_msgs.msg import Float64


def capture_images(pipeline):
	## Get frameset of color
	frames = pipeline.wait_for_frames()  
	color_frame = frames.get_color_frame()   # get color frame

	## Get the depth and color image and Convert images to numpy arrays
	color_image = np.asanyarray(color_frame.get_data())  # color image
	

	return color_image

def start_camera():
	""" Starts the camera """
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
		# exit(0)
			
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
			
	return pipeline, config
	

class CameraPublisher(Node):
	def __init__(self):
		super().__init__('minimal_publisher')
		self.publisher_ = self.create_publisher(Float64, 'topic', 10)

	def run(self):
		self.get_logger().info("Start run")
		camera_matrix = np.array([[654.68470569, 0.0, 309.89837988],
								  [0.0, 654.68470569, 177.32891715],
								  [0.0, 0.0, 1.0]])
		
		dist_coefficients = np.array(([[0.0, 0.0, 0.0, 0.0, 0.0]]))
		
		pipeline, config = start_camera()
		## Start streaming
		profile = pipeline.start(config)  # start the pipeline
		
		while True:
			img = capture_images(pipeline)
		
			# img = cv2.flip(img, 1)

			shape = img.shape[:2]
			new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefficients, shape, 0, shape)

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			_, gray = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
			
			aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
			parameters = aruco.DetectorParameters_create()
			cv2.undistort(img, camera_matrix, dist_coefficients, None, new_camera_mtx)

			corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
			self.get_logger().info(f'Marker detected, id is {ids}')
			
			if ids is None:
				pass
			else:
				
				for x in range(len(ids)):
					rotation_vector, translation_vector, _ = aruco.estimatePoseSingleMarkers(corners[x], 0.02, camera_matrix, dist_coefficients)
					(rotation_vector - translation_vector).any()  # get rid of that nasty numpy value array error

					for i in range(rotation_vector.shape[0]):
						aruco.drawAxis(gray, camera_matrix, dist_coefficients, rotation_vector[i, :, :],
									translation_vector[i, :, :], 0.02)
						aruco.drawAxis(img, camera_matrix, dist_coefficients, rotation_vector[i, :, :],
									translation_vector[i, :, :], 0.02)
						aruco.drawDetectedMarkers(gray, corners)
						aruco.drawDetectedMarkers(img, corners)
					
					mat = cv2.Rodrigues(rotation_vector)
					r = Rotation.from_matrix(mat[0])
					angle = r.as_euler('xyz', degrees=True)

					msg = Float64()
					msg.data = angle[2]
					self.publisher_.publish(msg)
					self.get_logger().info(f'Publishing: {msg.data}')
					# print(translation_vector[0][0], np.degrees(float(rotation_vector[0][0][0])))
					cv2.putText(img, f"The angle: {round(angle[2], 2)} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=3)
					
			# Display result frame
			cv2.imshow("frame", img)
			cv2.imshow("gray", gray)

			key = cv2.waitKey(1)
			if key == 27: # [esc] key
				break


def main(args=None):
	rclpy.init(args=args)

	camera_publisher = CameraPublisher()

	#rclpy.spin(camera_publisher)
	
	camera_publisher.run()
	
	camera_publisher.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
