# nmpc_state_node.py
#
# 1. Copy into px4_ros_ws/src/nmpc_state_node/
# 2. Run   chmod +x nmpc_state_node.py
# 3. Add an entry to your package.xml & CMakeLists.txt or just run with:
#      source ~/px4_ros_ws/install/setup.bash
#      python3 nmpc_state_node.py
#
# Requires:  rclpy  (installed with ROS 2)

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import qos_profile_sensor_data


from px4_msgs.msg import VehicleOdometry, SensorGyro

class NMPCStateNode(Node):
    def __init__(self):
        super().__init__('nmpc_state_node')

        self.state = np.zeros(13, dtype=np.float32)   # global buffer

        # --- subscribers ----------------------------------------------------
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_cb,
            qos_profile_sensor_data)      # queue size 50, enough for 200 Hz

        # --- (optional) timer to observe the vector -------------------------
        self.create_timer(0.1, self.debug_print)      # 10 Hz

    # ----------------- callbacks --------------------------------------------
    def odom_cb(self, msg: VehicleOdometry) -> None:
        self.state[0:3]   = msg.position           # x y z  (m)
        self.state[3:6]   = msg.velocity           # vx vy vz (m/s)
        self.state[6:10]  = msg.q                  # qx qy qz qw (unit)
        self.state[10:13]  = msg.angular_velocity           # wx wy wz (rad/s)

    def debug_print(self) -> None:
        """Print the vector every 0.1 s so you can watch it update."""
        np.set_printoptions(precision=3, suppress=True)
        self.get_logger().info(f'State: {self.state}')

# ------------------------- main --------------------------------------------
def main() -> None:
    print('begin')
    rclpy.init()
    node = NMPCStateNode()
    rclpy.spin(node)      # blocks until Ctrl-C
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()