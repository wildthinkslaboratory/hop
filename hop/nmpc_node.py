#  This is a new version of the controller that runs two seperate ROS2 nodes
#  The reason for trying this is that with a single node in python you can't have
#  Any parallel execution due to the GIL lock. That's the Global Interpreter Lock. 
#  I didn't know anything about this before I tried to multithread our previous controller node
#  and had disasterous results. Python can be so dumb.
#  Well the way around it is to run two separate nodes. This way the nmpc can execute in parallel
#  with the other monitoring and subscriptions needed for the drone. The other options are
#  to move the code to C++, or adapt the system equations to model the time delay that occurs from a single 
#  threaded controller that tries to do everything.
#
#
#  This node will collect a state reading from the main node hopefully every 20ms, it will 
#  run the nmpc and publish the control uninterrupted by the any other monitoring tasks.
#
#

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import ActuatorMotors, ActuatorServos
from hop_interfaces.msg import NMPCInput, NMPCStatus, CommandInput
from time import perf_counter
from datetime import datetime
from hop.utilities import output_data, quaternion_to_angle
from hop.integrators import RKSimulator
from hop.equations_of_motion import Equations6DOF

from casadi import DM
import numpy as np
from hop.constants import Constants
from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc

from hop.equations_of_motion import Equations6DOF
from hop.multiShootTDelay import DroneNMPCMultiShootTDelay
from collections import deque



from time import sleep
from random import uniform

mc = Constants()

from enum import Enum

class ShutdownReason(Enum):
    NONE = 0
    TIMEOUT = 1
    ANGLE_EXCEEDED = 2
    COMMAND = 3

class NMPCNode(Node):

    def __init__(self, dt = mc.dt):
        super().__init__('nmpc_node')
        
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )             
         

        ################# Subscriptons #########################  

        self.nmpc_input = self.create_subscription(
            NMPCInput, 
            '/hop/nmpc_input', 
            self.nmpc,
            qos_sub
        )

        self.command_input = self.create_subscription(
            CommandInput, 
            '/hop/Command_input', 
            self.command_callback,
            qos_sub
        )

        ############ Publishers #########################

        self.publisher_motors = self.create_publisher(
            ActuatorMotors, 
            '/fmu/in/actuator_motors', 
            qos_pub
        )

        self.publisher_servos = self.create_publisher(
            ActuatorServos, 
            '/fmu/in/actuator_servos', 
            qos_pub
        )

        self.publisher_status = self.create_publisher(
            NMPCStatus, 
            '/hop/nmpc_status', 
            qos_pub
        )

        ####################  locally store data ###################

        self.log_rows = []
        self.waypoint_i = 0
        self.command = CommandInput.NONE
        self.q = np.array([0.0, 0.0, 0.0, 1.0])

        self.equations = Equations6DOF(mc)
        self.rk_sim = RKSimulator(0.02, 1) # set up to make one 20ms step
        self.control_history = deque(
            [np.array([0.0, 0.0, mc.hover_thrust, 0.0]) for i in range(mc.nmpc_delay)],
            maxlen=mc.nmpc_delay
        )

        self.model = DroneModel(mc)
        self.mpc = DroneNMPCdompc(mc.dt, self.model.model)
        self.mpc.setup_cost()
        self.mpc.set_start_state(mc.x0)

        self.shutdown_reason = ShutdownReason.NONE
        self.first_nmpc_call = True
        status = NMPCStatus()
        status.status = NMPCStatus.READY
        status.timestamp = self.get_clock().now().nanoseconds // 1000
        self.publisher_status.publish(status)
        self.get_logger().info('NMPC Ready')
        self.start_time = perf_counter()


############################# callbacks  ####################################

    
    def nmpc(self, msg):

        if self.first_nmpc_call:
            self.first_nmpc_call = False
            self.start_time = perf_counter()

        nmpc_receive_time = self.get_clock().now().nanoseconds // 1000
        runtime = perf_counter() - self.start_time
        x_theta, y_theta, theta = quaternion_to_angle(self.q)

        if runtime > mc.timelimit:
            self.shutdown_reason = ShutdownReason.TIMEOUT
        elif theta > mc.shutdown_angle:
            self.shutdown_reason = ShutdownReason.ANGLE_EXCEEDED

        if not self.shutdown_reason == ShutdownReason.NONE:
            self.shutdown()
        else:
            start_time = perf_counter()
            state = DM(np.array(msg.state))
            raw_state = DM(state)
            parameters = mc.waypoints[self.waypoint_i]
            parameters[3] = msg.filtered_voltage   
            self.q = np.reshape(state[6:10], (4,))
            control = np.array([0.0, 0.0, 0.0, 0.0])
            if mc.run_nmpc:    
                
                # integrate the state forward with the control history before calling the nmpc
                for control in self.control_history:
                    state = self.rk_sim.make_step(self.equations.f, state, control, parameters)

                self.mpc.set_waypoint(parameters)
                control = np.array(self.mpc.mpc.make_step(state)).flatten()
                self.control_history.append(control.copy())
  

                

            pwm_servos, pwm_motors = self.control_translator(control)   
            self.run_motors(pwm_motors)
            self.run_servos(pwm_servos)   

            nmpc_time = perf_counter() - start_time

            self.log_rows.append({
                'state': state.full().flatten().tolist(),
                'raw_state': raw_state.full().flatten().tolist(),
                'control': control.tolist(),
                'timing': [msg.timestamp_sample, msg.main_receive_time, msg.main_send_time, nmpc_receive_time, nmpc_time],
                'pwm_motors': pwm_motors,
                'pwm_servos': pwm_servos,
                'parameters': parameters.tolist(),
                'current_a': msg.current_a,
                'current_average_a': msg.current_average_a,
                'discharged_mah': msg.discharged_mah,
                'remaining': msg.remaining,
                'raw_voltage': msg.battery_voltage,
            })
    

    def command_callback(self, msg):
        if msg.command == CommandInput.INC_WAYPOINT:
            self.waypoint_i = self.waypoint_i + 1
        else:
            self.command = msg.command
            self.get_logger().info('Command received ' + str(msg.command))
            if msg.command == CommandInput.SHUTDOWN:
                self.shutdown_reason = ShutdownReason.COMMAND
                self.shutdown()



    def get_angle_pwm(self, gimbal_angles):
        gimbal_angles[0] = gimbal_angles[0] + mc.gimbal_offset[0]      # gimbal offset
        gimbal_angles[0] = np.clip(gimbal_angles[0], mc.outer_gimbal_range[0], mc.outer_gimbal_range[1])
        gimbal_angles[1] = np.clip(gimbal_angles[1],  mc.inner_gimbal_range[0], mc.inner_gimbal_range[1])

        outer_angle_pwm = gimbal_angles[0] / mc.gmb_deg_1pwm
        inner_angle_pwm = gimbal_angles[1] / mc.gmb_deg_1pwm
        
        return outer_angle_pwm, inner_angle_pwm
    
    def get_thrust_pwm(self, thrust_values):
        top_prop_thrust = thrust_values[0] - thrust_values[1]/2
        bottom_prop_thrust = thrust_values[0] + thrust_values[1]/2
        top_prop_pwm = np.clip(top_prop_thrust, 0, 1)
        bottom_prop_pwm = np.clip(bottom_prop_thrust, 0, 1)
        return top_prop_pwm, bottom_prop_pwm

    def control_translator(self, control):
        gimbal_angles = control[0:2].copy()
        thrust_values = control[2:4].copy()
        outer_angle_pwm, inner_angle_pwm = self.get_angle_pwm(gimbal_angles)
        top_prop_pwm, bottom_prop_pwm = self.get_thrust_pwm(thrust_values)

        return [outer_angle_pwm, inner_angle_pwm], [top_prop_pwm, bottom_prop_pwm]
        

    def shutdown(self):

        # shutdown motors and servos
        self.run_motors([0.0, 0.0])
        self.run_servos([0.0, 0.0])

        # tell the main control node to disarm and shutdown
        disarm_msg = NMPCStatus()
        disarm_msg.status = NMPCStatus.DISARM_REQUEST
        self.publisher_status.publish(disarm_msg)
      
        self.get_logger().info('NMPC Node shutting down: ' + str(self.shutdown_reason))

        # write out the nmpc data
        data = {'constants': mc.__dict__(), 'run_data': self.log_rows}
        output_data(data, "src/hop/plotter_logs/current.json")
        formatted_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        output_data(data, "src/hop/plotter_logs/" + formatted_date + "log.json")
        rclpy.shutdown()


    # # when we exit do clean up and output the run data
    # def destroy_node(self):
    #     super().destroy_node()

################################### PUBLISHER functions #######################################



    def run_motors(self, pwm_motors):
        motor_command = ActuatorMotors()
        t = self.get_clock().now().nanoseconds // 1000
        motor_command.timestamp_sample = t
        motor_command.timestamp = t
        motor_command.control = [pwm_motors[0], pwm_motors[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 4 motors + 8 unused
        
        self.publisher_motors.publish(motor_command)
        if self.logging_on:
            self.get_logger().info('Publishing motor pwm ' + str(pwm_motors))


    def run_servos(self, pwm_servos):
        servo_command = ActuatorServos()
        t = self.get_clock().now().nanoseconds // 1000
        servo_command.timestamp_sample = t
        servo_command.timestamp = t
        servo_command.control = [-pwm_servos[0], -pwm_servos[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 4 motors + 4 unused

        self.publisher_servos.publish(servo_command)
        if self.logging_on:
            self.get_logger().info('Publishing servo pwm ' + str(pwm_servos))


def main(args=None):
    rclpy.init(args=args)
    nmpc = NMPCNode()
    nmpc.logging_on = False

    try:
        rclpy.spin(nmpc)
    except SystemExit:
        pass
    finally:
        nmpc.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()




