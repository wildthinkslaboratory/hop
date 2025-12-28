import rclpy
from hop.offboard_node import OffBoardNode
from hop.constants import Constants
mc = Constants()

class TestMotors(OffBoardNode):

    def __init__(self):
        super().__init__('test_motors', timelimit=100, dt=mc.dt)


    def timer_callback(self):

        # manage key presses
        if self.key == 'u':
            self.key = ''
            self.pwm_motors[0] += 0.1
            self.pwm_motors[1] += 0.1
            self.get_logger().info('motor pwm ' + str(self.pwm_motors))

        elif not self.key == '':
            self.pwm_motors = [0.0, 0.0]
            raise SystemExit
        
        super().timer_callback()
    
        

def main(args=None):
    rclpy.init(args=args)
    motor_test = TestMotors()
    motor_test.logging_on = False

    try:
        rclpy.spin(motor_test)
    except SystemExit:
        pass
    finally:
        motor_test.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()