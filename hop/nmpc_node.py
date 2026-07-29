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


from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import ActuatorMotors, ActuatorServos
from time import perf_counter
from datetime import datetime

from casadi import DM
import numpy as np
from hop.constants import Constants
from hop.utilities import output_data

mc = Constants()



class NMPCNode(Node):

    def __init__(self, name, timelimit = None, dt = mc.dt):
        super().__init__(name)

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

        # it will have one state subscription from main node



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

        ####################  locally store data ###################

        self.log_rows = []



############################# callbacks  ####################################

    # publish all of our messages
    def nmpc(self, msg):
        state = msg.state
        state_sample_time = msg.state_sample_time
        parameters = msg.parameters       # this needs to have latest voltage     

        self.mpc.set_waypoint(parameters)
        control = np.array(self.mpc.mpc.make_step(state)).flatten()
        pwm_servos, pwm_motors = self.control_translator(control)   

        self.run_motors(pwm_motors)
        self.run_servos(pwm_servos)   
        
        self.log_rows.append({
            'state': state.full().flatten().tolist(),
            'control': control.tolist(),
            'timing': [state_sample_time, 0.0],
            'pwm_motors': pwm_motors,
            'pwm_servos': pwm_servos,
            'voltage': parameters[3],
            'parameters': parameters.tolist(),
            'timestamp' : 0.0
        })
    
    
    def get_angle_pwm(self, gimbal_angles):
        gimbal_angles[0] = gimbal_angles[0]       # gimbal offset
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
        gimbal_angles = control[0:2]
        thrust_values = control[2:4]
        outer_angle_pwm, inner_angle_pwm = self.get_angle_pwm(gimbal_angles)
        top_prop_pwm, bottom_prop_pwm = self.get_thrust_pwm(thrust_values)

        return [outer_angle_pwm, inner_angle_pwm], [top_prop_pwm, bottom_prop_pwm]
        

    # when we exit do clean up and output the run data
    def destroy_node(self):

        # write out the nmpc data
        data = {'constants': mc.__dict__(), 'run_data': self.log_rows}
        output_data(data, "src/hop/plotter_logs/current.json")
        formatted_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        output_data(data, "src/hop/plotter_logs/" + formatted_date + "log.json")
        super().destroy_node()

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
        self.timing_data[5] = t 
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()





