import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import BatteryStatus, OffboardControlMode, VehicleStatus, VehicleCommand, VehicleOdometry
from hop_interfaces.msg import NMPCInput, NMPCStatus

from casadi import DM
import numpy as np
from hop.constants import Constants
from hop.utilities import output_data, quaternion_multiply
from datetime import datetime
from math import sqrt
mc = Constants()

# this is all needed for keyboard input
import sys
import select
import termios
import tty
import threading

class ControlNode(Node):

    def __init__(self, dt = mc.dt):
        super().__init__('main_control_node')

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
      
        self.vehicle_odometry = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.state_callback,
            qos_sub
        )

        self.vehicle_status = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_sub
        )

        self.battery_status = self.create_subscription(
            BatteryStatus,
            '/fmu/out/battery_status',
            self.battery_callback,
            qos_sub
        )

        self.nmpc_status = self.create_subscription(
            NMPCStatus,
            '/hop/nmpc_status',
            self.nmpc_status_callback,
            qos_sub
        )

        ############ Publishers #########################

        self.publisher_vehicle_command = self.create_publisher(
            VehicleCommand, 
            '/fmu/in/vehicle_command', 
            qos_pub
        )

        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, 
            '/fmu/in/offboard_control_mode', 
            qos_pub
        )

        self.publisher_nmpc_input = self.create_publisher(
            NMPCInput, 
            '/hop/nmpc_input', 
            qos_pub
        )

        ####################  locally store data ###################
        
        # SHUTDOWN move waypoint_i to nmpc node

        self.dt = dt
        self.logging_on = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.key = ''
        self.waypoint_i = 0
        self.first_arm = True
        self.state = mc.x0
        self.voltage = 0.0    
        self.timestamp_sample = 0
        self.main_receive_time = 0  
        self.nmpc_status = NMPCStatus.STARTING          
        self.armed = False
        self.count = 0
        self.x_offset = 0.0  # offsets needed for optical flow
        self.y_offset = 0.0

        self.stop_requested = False

        # SHUTDOWN strip out the keyboard code 

        # set up for keyboard reading
        if not sys.stdin.isatty():
            raise RuntimeError("The node must run in an interactive terminal.")
        self.std_in_fd = sys.stdin.fileno()
        self.term_settings = termios.tcgetattr(self.std_in_fd)
        tty.setcbreak(self.std_in_fd)


        self.timer = self.create_timer(self.dt, self.main_loop)    


#############################  ####################################


    # 1. forward state to nmpc
    # 2. maintain offboard mode
    # manage arming at beginning of flight and disarming when stop requested
    def main_loop(self):

        # SHUTDOWN strip out keyboard code

        # monitor keyboard presses
        if select.select([sys.stdin], [], [], 0.0)[0]:
            self.key = sys.stdin.read(1)
            if self.logging_on:
                self.get_logger().info(f"Key pressed: {self.key}")


        # switch this to look at nmpc_status MOTOR_SHUTDOWN OR LAND
        if not self.key == '': # any key press means stop the run
            if self.armed:
                self.disarm()
            else:
                rclpy.shutdown()
        else: 
            if self.armed and self.nmpc_status == NMPCStatus.READY:
                self.publish_nmpc_input()


            self.count += 1
            self.maintain_offboard()

            if not self.armed and self.count >= 100 and self.count % 50 == 0:
                self.offboard_arm()



        





################################### PUBLISHER functions #######################################

    def maintain_offboard(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.direct_actuator = True
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.publisher_offboard_mode.publish(offboard_msg)


    def offboard_arm(self):
        # switch to Offboard
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
        # arm
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        if self.logging_on:
            self.get_logger().info('Publishing arm and offboard mode commands')


    def publish_vehicle_command(self, command, p1=0., p2=0.):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.command = command
        msg.param1, msg.param2 = float(p1), float(p2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 2
        msg.source_component = 1
        msg.from_external = True
        self.publisher_vehicle_command.publish(msg)


    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            0.0,          # param1 = disarm
            21196.0       # param2 = FORCE (disarm in-air)
        )


    def publish_nmpc_input(self):
        msg = NMPCInput()
        msg.timestamp_sample = self.timestamp_sample
        msg.main_receive_time = self.main_receive_time
        msg.main_send_time = self.get_clock().now().nanoseconds // 1000
        msg.state = self.state

        # SHUTDOWN move waypoint_i to nmpc node just send the voltage
        # update the message nmpc_input

        # fill in the parameters
        parameters = [0.0] * 8
        parameters[:5] = mc.waypoints[self.waypoint_i]
        parameters[3] = self.voltage
        
        msg.parameters = parameters
        self.publisher_nmpc_input.publish(msg)


############################# subscription callbacks  ####################################

    # recieve armed status
    def vehicle_status_callback(self, msg):
        was_armed = self.armed
        if msg.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            self.armed = True
            if self.first_arm:
                self.first_arm = False
                self.get_logger().info('Vehicle is ARMED')
        elif msg.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
            self.armed = False
            self.get_logger().info('Vehicle is DISARMED')

        # if it switches from disarmed to armed then we set the x and y offsets
        if not was_armed and self.armed:
            self.x_offset = float(self.state[0])
            self.y_offset = float(self.state[1])

    # recieve armed status
    def battery_callback(self, msg):
        self.voltage = msg.voltage_v
        

    def nmpc_status_callback(self, msg):
        self.nmpc_status = msg.status
        self.get_logger().info('NMPC Ready Received')


    # recieve vehicle odometry message
    def state_callback(self, msg):
        self.state = [0.0] * 13

        # px4 uses NED (North, East, Down) for position, 
        #  quaternion (w, i, j, k) gives rotation from body frame FRD (front, right, down) 
        # to NED and angular volocity in body FRD.
        # 
        # The drone body frame is in a FLU (front, left, up) orientation 
        # and is rotated 90 degrees clockwise looking down the up z axis
        # from the px4 body frame. 
        # 

        # position is translated from NED to ENU
        pos = np.array(msg.position)
        vel = np.array(msg.velocity)
        self.state[0:3] = [pos[1] - self.x_offset, pos[0] - self.y_offset, -pos[2]]
        self.state[3:6] = [vel[1], vel[0], -vel[2]]
    
        # Front Left Up to Front Right Down translation (w, x, y, z)
        FLU_FRD = np.array([0, sqrt(2)/2, sqrt(2)/2, 0])

        # North East Down to East North Up translation (w, x, y, z)
        NED_ENU = np.array([0, -sqrt(2)/2, -sqrt(2)/2, 0])
        
        q = np.array(msg.q) # incoming px4 quaternion is in body FRU to world NED

        # build quaternion FLU_FRD * FRD_NED * NED_ENU = FLU_ENU
        q_FLU_ENU = quaternion_multiply(FLU_FRD, quaternion_multiply(q, NED_ENU))

        # renorm and translate from (w, i, j, k) to (i, j, k, w) form.
        norm = np.linalg.norm(q_FLU_ENU)
        if norm > 0:
            q_FLU_ENU /= norm
        self.state[6:10] = np.array([q_FLU_ENU[1], q_FLU_ENU[2], q_FLU_ENU[3], q_FLU_ENU[0]])
               
        ang_vel = msg.angular_velocity
        self.state[10:13] = [ang_vel[1], ang_vel[0], -ang_vel[2]]

        self.timestamp_sample = msg.timestamp_sample
        self.main_receive_time = self.get_clock().now().nanoseconds // 1000
       
        if self.logging_on:
            self.get_logger().info(
                f"""\n=== NMPC Step ===
                State:
                p: {self.state[0:3]}
                v: {self.state[3:6]}
                q: {self.state[6:10]}
                w: {self.state[10:13]}
                """
            )


    ############################# other functions  ####################################


    # when we exit do clean up and output the run data
    def destroy_node(self):

        # reset terminal settings
        termios.tcsetattr(self.std_in_fd, termios.TCSANOW, self.term_settings)
        super().destroy_node()




def main(args=None):
    rclpy.init(args=args)
    main_node = ControlNode()
    main_node.logging_on = False

    try:
        rclpy.spin(main_node)
    except SystemExit:
        pass
    finally:
        main_node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()