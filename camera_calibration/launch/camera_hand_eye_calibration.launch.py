from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    camera_calibration_goals = PathJoinSubstitution(
        [FindPackageShare("camera_calibration"), "config", "camera_calibration_config.yaml"]
    )

    return LaunchDescription(
        [
            Node(
                package="ros2_control_test_nodes",
                executable="publisher_camera_calibration_ik_controller",
                name="publisher_camera_calibration_ik_controller",
                parameters=[camera_calibration_ur5_goals],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },
             )
             Node(
                package="camera_calibration",
                executable="timer_node_image_capture",
                name="timer_node_image_capture",
                parameters=[camera_calibration_goals],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },
            )
        ]
    )
