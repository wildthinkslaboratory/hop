import rclpy
from hop.offboard_node import OffBoardNode
from hop.constants import Constants
from gpiozero import LED
mc = Constants()
from numpy import clip  

class TestServos(OffBoardNode):

    def __init__(self):
        self.run_simple_delay_test = False
        self.run_ramp_delay_test = False
        self.led = LED(27)
        self.ramp_i = 0
        self.ramp = [0, 4, 8, 12, 16, 20, 16, 12, 8, 4, 0, -4, -8, -12, -16, -20, -16, -8, -4, 
                     0, 4, 8, 12, 16, 20, 16, 12, 8, 4, 0, -4, -8, -12, -16, -20, -16, -8, -4, 
                     0, 4, 8, 12, 16, 20, 16, 12, 8, 4, 0, -4, -8, -12, -16, -20, -16, -8, -4, 
                     0, 4, 8, 12, 16, 20, 16, 12, 8, 4, 0, -4, -8, -12, -16, -20, -16, -8, -4, 0]
        
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

        elif self.key == 't':
            self.key = ''
            self.run_simple_delay_test = True
        elif self.key == 'r':
            self.key = ''
            self.run_ramp_delay_test = True
        elif not self.key == '':
            raise SystemExit
        
        super().timer_callback()
    
    
    def run_servos(self):

        if self.run_simple_delay_test:
            self.ramp_i += 1
            if self.ramp_i % 100 < 50:
                self.led.on()
                self.pwm_servos = [6.0 / mc.gmb_deg_1pwm, 0.0]
            else:
                self.led.off()
                self.pwm_servos = [0.0, 0.0]


        elif self.run_ramp_delay_test:
            if self.ramp_i < len(self.ramp)-1:
                self.ramp_i += 1
                self.pwm_servos =  [self.ramp[self.ramp_i] / mc.gmb_deg_1pwm, 0.0]
                if self.ramp[self.ramp_i-1] < self.ramp[self.ramp_i]:
                    self.led.on()
                else:
                    self.led.off()

        super().run_servos()
        

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