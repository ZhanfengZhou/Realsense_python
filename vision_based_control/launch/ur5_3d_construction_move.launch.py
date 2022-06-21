from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():


    online_camera_calibration_goals = PathJoinSubstitution(
        [FindPackageShare("camera_calibration"), "config", "ur5_3d_construction_move.yaml"]
    )


    return LaunchDescription(
        [
            Node(
                package="ros2_control_test_nodes",
                executable="publisher_camera_calibration_ik_controller",
                name="publisher_ur5_3d_construction_move_controller",
                parameters=[online_camera_calibration_goals],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },

#             Node(
#                package="camera_calibration",
#                executable="timer_node_image_capture",
#                name="timer_node_image_capture",
#                parameters=[online_camera_calibration_goals],
#                output={
#                    "stdout": "screen",
#                    "stderr": "screen",
#                },

            )
        ]
    )
