import rclpy
from hop.offboard_node import OffBoardNode
from hop.constants import Constants
mc = Constants()

class TestServos(OffBoardNode):

    def __init__(self):
        super().__init__('test_servos', timelimit=100, dt=mc.dt)


    def timer_callback(self):

        # manage key presses
        if self.key == 'u':
            self.key = ''
            self.pwm_servos[0] += 0.1
            self.pwm_servos[1] += 0.1
            self.get_logger().info('servo pwm ' + str(self.pwm_servos))

            
        elif self.key == 'j':
            self.key = ''
            self.pwm_servos[0] -= 0.1
            self.pwm_servos[1] -= 0.1
            self.get_logger().info('motor pwm ' + str(self.pwm_servos))

        elif not self.key == '':
            raise SystemExit
        
        super().timer_callback()
    


        

def main(args=None):
    rclpy.init(args=args)
    servo_test = TestServos()
    servo_test.logging_on = False

    try:
        rclpy.spin(servo_test)
    except SystemExit:
        pass
    finally:
        servo_test.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()