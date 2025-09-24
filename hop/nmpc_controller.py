import rclpy
import numpy as np
import casadi as ca
from casadi import sin, cos
import do_mpc

from hop.drone_model import DroneModel
from hop.drone_mpc import DroneMPC
from hop.offboard_node import OffBoardNode
from hop.constants import Constants
mc = Constants()

class NMPC(OffBoardNode):

    def __init__(self):
        super().__init__('nmpc_controller', timelimit=100, dt=mc.dt)

        self.model = DroneModel()
        self.mpc = DroneMPC(mc.dt, self.model.model)
        self.mpc.set_goal_state()
        self.mpc.set_start_state(mc.x0)

    def timer_callback(self):

        # quit if someone pressed a key
        if not self.key == '':
            raise SystemExit


        # for attitude testing we are turning off the position data and velocity
        self.state[0] = 0.0
        self.state[1] = 0.0
        self.state[2] = 1.0
        self.state[3] = 0.0
        self.state[4] = 0.0
        self.state[5] = 0.0

        control = self.mpc.mpc.make_step(self.state)
        self.control = np.array(control).flatten()
        self.control_translator()   

        # for attitude testing set both thrusters to 0.1 pwm
        self.pwm_motors =  [0.5, 0.5]

        super().timer_callback()
    
    
    def get_angle_pwm(self, gimbal_angles):
        gimbal_angles[0] = gimbal_angles[0]       # gimbal offset
        gimbal_angles[0] = np.clip(gimbal_angles[0], mc.outer_gimbal_range[0], mc.outer_gimbal_range[1])
        gimbal_angles[1] = np.clip(gimbal_angles[1],  mc.inner_gimbal_range[0], mc.inner_gimbal_range[1])

        outer_angle_pwm = gimbal_angles[0] / 43

        inner_angle_pwm = gimbal_angles[1] / 43
        return outer_angle_pwm, inner_angle_pwm
    
    def get_thrust_pwm(self, thrust_values):
        top_prop_thrust = thrust_values[0] + thrust_values[1]/2
        bottom_prop_thrust = thrust_values[0] - thrust_values[1]/2
        top_prop_pwm = top_prop_thrust / mc.prop_thrust_constraint
        bottom_prop_pwm = bottom_prop_thrust / mc.prop_thrust_constraint
        top_prop_pwm = np.clip(top_prop_pwm, 0, 1)
        bottom_prop_pwm = np.clip(bottom_prop_pwm, 0, 1)
        return top_prop_pwm, bottom_prop_pwm

    def control_translator(self):
        gimbal_angles = self.control[0:2]
        thrust_values = self.control[2:4]
        outer_angle, inner_angle = self.get_angle_pwm(gimbal_angles)
        top_prop_pwm, bottom_prop_pwm = self.get_thrust_pwm(thrust_values)
        self.pwm_servos =  [outer_angle, inner_angle]
        self.pwm_motors =  [top_prop_pwm, bottom_prop_pwm]
        

def main(args=None):
    rclpy.init(args=args)
    nmpc = NMPC()
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