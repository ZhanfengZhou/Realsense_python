from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():


    vision_based_grasp_objects_goals = PathJoinSubstitution(
        [FindPackageShare("vision_based_control"), "config", "ur5_human_robot_interactive_grasp_yolo.yaml"]
    )


    return LaunchDescription(
        [
            Node(
                package="ros2_control_test_nodes",
                executable="node_human_robot_interactive_grasp_yolo",
                name="human_robot_interactive_grasp_yolo",
                parameters=[vision_based_grasp_objects_goals],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },

            )
        ]
    )
