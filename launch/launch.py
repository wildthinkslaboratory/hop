from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hop',
            executable='dynamics',
            name='dynamics'
        ),
        Node(
            package='hop',
            executable='nmpc',
            name='nmpc'
        )
    ])