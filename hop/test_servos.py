import rclpy
from hop.offboard_node import OffBoardNode
from hop.constants import Constants
mc = Constants()

class MotorTest(OffBoardNode):

    def __init__(self):
        super().__init__('nmpc_controller', timelimit=100, dt=mc.dt)


    def timer_callback(self):

        # manage key presses
        if self.key == 'u':
            self.pwm_servos[0] += 0.1
            self.pwm_servos[1] += 0.1
            if self.logging_on:
                self.get_logger().info('motor pwm ' + str(self.pwm_motors))
            
        elif self.key == 'j':
            self.pwm_servos[0] -= 0.1
            self.pwm_servos[1] -= 0.1
            if self.logging_on:
                self.get_logger().info('motor pwm ' + str(self.pwm_motors))

        elif not self.key == '':
            raise SystemExit
        
        super().timer_callback()
    


        

def main(args=None):
    rclpy.init(args=args)
    nmpc = MotorTest()
    nmpc.logging_on = True

    try:
        rclpy.spin(nmpc)
    except SystemExit:
        pass
    finally:
        nmpc.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()