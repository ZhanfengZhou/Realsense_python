from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():


    vision_based_grasp_objects_goals = PathJoinSubstitution(
        [FindPackageShare("vision_based_control"), "config", "ur5_vision_based_grasp_marker_from_human.yaml"]
    )


    return LaunchDescription(
        [
            Node(
                package="ros2_control_test_nodes",
                executable="node_human_robot_interactive_grasp_with_marker",
                name="vision_based_grasp_marker_from_human",
                parameters=[vision_based_grasp_objects_goals],
                output={
                    "stdout": "screen",
                    "stderr": "screen",
                },

            )
        ]
    )
