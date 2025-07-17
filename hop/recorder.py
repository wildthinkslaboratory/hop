import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from hop.constants import Constants

mc = Constants()

class Recorder(Node):

    def __init__(self):
        super().__init__('recorder')

        self.nmpc_sub = self.create_subscription(
            Float64MultiArray,
            'NMPC',
            self.nmpc_callback,
            10)
        self.nmpc_sub

        self.dynamics_sub = self.create_subscription(
            Float64MultiArray,
            'dynamics',
            self.dynamics_callback,
            10)
        self.dynamics_sub

        self.recordfile = open('recordings/record.json', 'w')
        



  
    def nmpc_callback(self, msg):
        self.get_logger().info('nmpc: "%s"' % msg.data)
        

    def dynamics_callback(self, msg):
        self.get_logger().info('dynamics: "%s"' % msg.data)


    



def main(args=None):
    print('recording data')
    rclpy.init(args=args)

    recorder = Recorder()

    rclpy.spin(recorder)
    recorder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

