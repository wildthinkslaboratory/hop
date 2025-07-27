import rclpy
from hop.offboard_node import OffBoardNode

class ServoTest(OffBoardNode):

    def __init__(self):
        super().__init__('ServoTestNode', timelimit=2)
        

def main(args=None):
    rclpy.init(args=args)
    servo_test = ServoTest()

    try:
        rclpy.spin(servo_test)
    except SystemExit:
        servo_test.finalize()
    finally:
        servo_test.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()