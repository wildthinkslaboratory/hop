from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hop',
            executable='nmpc_controller',
            name='nmpc_controller'
        )
        ,
        # Node(
        #     package='hop',
        #     executable='test_servos',
        #     name='test_servos'
        # )
    ])
    