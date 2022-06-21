from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

#    camera_calibration_ur5_goals = PathJoinSubstitution(
#        [FindPackageShare("camera_calibration"), "config", "camera_calibration_config.yaml"]
#    )
    vision_guided_grasp_goals = PathJoinSubstitution(
        [FindPackageShare("camera_calibration"), "config", "ur5_object_grasping_position_test.yaml"]
    )


    return LaunchDescription(
        [
            Node(
                package="ros2_control_test_nodes",
                executable="publisher_vision_guided_grasp_controller",
                name="publisher_ur5_3d_construction_move_controller",
                parameters=[vision_guided_grasp_goals],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },

            )
        ]
    )

