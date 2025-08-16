import rclpy
from hop.offboard_node import OffBoardNode

class ServoTest(OffBoardNode):

    def __init__(self):
        super().__init__('ServoTestNode', timelimit=100)

    def timer_callback(self):
        self.pwm_servos = [1, 1]

        # quit if someone pressed a key
        if not self.key == '':
            raise SystemExit

        super().timer_callback()



def main(args=None):
    rclpy.init(args=args)
    servo_test = ServoTest()
    # servo_test.logging_on = False
    try:
        rclpy.spin(servo_test)
    except SystemExit:
        pass
    finally:
        servo_test.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()