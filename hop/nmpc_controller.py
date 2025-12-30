import rclpy
import numpy as np
import casadi as ca
from casadi import sin, cos
import do_mpc

from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc
from hop.offboard_node import OffBoardNode
from hop.constants import Constants
mc = Constants()

class NMPC(OffBoardNode):

    def __init__(self):
        super().__init__('nmpc_controller', timelimit=100, dt=mc.dt)

        self.model = DroneModel(mc)
        self.mpc = DroneNMPCdompc(mc.dt, self.model.model)
        self.mpc.setup_cost()
        self.mpc.set_start_state(mc.x0)

        self.waypoint_i = 0
        self.nmpc_on = False

    def timer_callback(self):

        # key press controls
        # u for next way point
        # l for land
        # any other key to cut motors
        if self.key == 'u':
            self.key = ''
            self.waypoint_i += 1
            self.get_logger().info('new waypoint ' + str(mc.waypoints[self.waypoint_i][:3]))
        elif self.key == 'l':
            self.key = ''
            land = np.array([self.state[0], self.state[1], 0.0])
            self.mpc.set_waypoint(mc.land)
            self.get_logger().info('landing ' + str(land))
        elif not self.key == '':
            self.pwm_motors = [0.0, 0.0]
            self.run_motors()
            raise SystemExit

        if self.armed:
            mc.waypoints[self.waypoint_i][3] = self.voltage
            self.mpc.set_waypoint(mc.waypoints[self.waypoint_i])
            control = self.mpc.mpc.make_step(self.state)
            self.control = np.array(control).flatten()
            self.control_translator()   
            
        super().timer_callback()
    
    
    def get_angle_pwm(self, gimbal_angles):
        gimbal_angles[0] = gimbal_angles[0]       # gimbal offset
        gimbal_angles[0] = np.clip(gimbal_angles[0], mc.outer_gimbal_range[0], mc.outer_gimbal_range[1])
        gimbal_angles[1] = np.clip(gimbal_angles[1],  mc.inner_gimbal_range[0], mc.inner_gimbal_range[1])

        outer_angle_pwm = gimbal_angles[0] / 43
        inner_angle_pwm = gimbal_angles[1] / 43
        
        return outer_angle_pwm, inner_angle_pwm
    
    def get_thrust_pwm(self, thrust_values):
        top_prop_thrust = thrust_values[0] - thrust_values[1]/2
        bottom_prop_thrust = thrust_values[0] + thrust_values[1]/2
        top_prop_pwm = np.clip(top_prop_thrust, 0, 1)
        bottom_prop_pwm = np.clip(bottom_prop_thrust, 0, 1)
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
    nmpc.logging_on = False

    try:
        rclpy.spin(nmpc)
    except SystemExit:
        pass
    finally:
        nmpc.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()