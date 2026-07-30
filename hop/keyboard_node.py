import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from hop_interfaces.msg import NMPCInput, NMPCStatus, KeyboardInput


import sys
import select
import termios
import tty
import threading

class KeyboardMonitorNode(Node):

    def __init__(self, dt = 0.02):
        super().__init__('keyboard_node')
        self.dt = dt

        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )          

        self.publisher_keyboard_input = self.create_publisher(
            KeyboardInput, 
            '/hop/Keyboard_input', 
            qos_pub
        )
        
        self.key = ''

        # set up for keyboard reading
        if not sys.stdin.isatty():
            raise RuntimeError("The node must run in an interactive terminal.")
        self.std_in_fd = sys.stdin.fileno()
        self.term_settings = termios.tcgetattr(self.std_in_fd)
        tty.setcbreak(self.std_in_fd)

        self.timer = self.create_timer(self.dt, self.keyboard_callback)    
        self.get_logger().info('Keyboard Monitor Node Running')
        

    # give it a key node shutdown
    # only publish if the key changes
    def keyboard_callback(self):

        # monitor keyboard presses
        if select.select([sys.stdin], [], [], 0.0)[0]:
            self.key = sys.stdin.read(1)
            
        # check to see if a key was pressed
        if not self.key == '':

            if self.key == 's': # s means shutdown the node
                self.get_logger().info('shutting down keyboard node')
                rclpy.shutdown()
            else:
                msg = KeyboardInput()
                if self.key == 'u':
                    msg.command = KeyboardInput.INC_WAYPOINT
                    self.get_logger().info('Increment waypoint requested')
                elif self.key == 'l':
                    msg.command = KeyboardInput.LAND
                    self.get_logger().info('LAND requested')
                elif not self.key == '':
                    msg.command = KeyboardInput.SHUTDOWN
                    self.get_logger().info('SHUTDOWN requested')

                # reset the key
                self.key = ''

                msg.timestamp = self.get_clock().now().nanoseconds // 1000
                self.publisher_keyboard_input.publish(msg)




    def destroy_node(self):
        # reset terminal settings
        termios.tcsetattr(self.std_in_fd, termios.TCSANOW, self.term_settings)
        super().destroy_node()




def main(args=None):
    rclpy.init(args=args)
    keyboard_node = KeyboardMonitorNode()

    try:
        rclpy.spin(keyboard_node)
    except SystemExit:
        pass
    finally:
        keyboard_node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()