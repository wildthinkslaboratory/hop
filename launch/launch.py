from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hop',
            executable='main_controller',
            name='main_controller',
            output='screen',
        )
        ,
        Node(
            package='hop',
            executable='nmpc_node',
            name='nmpc_node',
            output='screen',
        )
    ])
    
