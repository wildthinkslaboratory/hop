#  This is a new version of the controller that runs three seperate ROS2 nodes
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
#  nmpc_node :     This node will collect a state reading from the main node every 20ms, it will 
#                  run the nmpc and publish the control uninterrupted by the any other monitoring tasks.
#
#  keyboard_node : This node monitors the keyboard for navigation controls, incrementing waypoints, landing
#                  and and shutdown
#
# main_control_node : This node does everything else. Subscribes to state odometry, battery status and vehicle
#                  status. Takes care of arming and disarming. Maintains offboard mode.
#
#


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import BatteryStatus, OffboardControlMode, VehicleStatus, VehicleCommand, VehicleOdometry
from hop_interfaces.msg import NMPCInput, NMPCStatus
from casadi import DM
import numpy as np
from hop.constants import Constants
from hop.utilities import quaternion_multiply
from math import sqrt
from collections import deque

mc = Constants()

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

        # read this data from pixhawk and then we translate it to 
        # the appropriate coordinate systems and forward to the nmpc_node
        self.state = mc.x0

        # battery status data for thrust analysis
        self.voltage = 0.0  
        self.filtered_voltage = 0.0  
        self.voltage_history = deque(maxlen=12)
        self.current_a = 0.0
        self.current_average_a = 0.0
        self.discharged_mah = 0.0
        self.remaining = 0.0

        self.timestamp_sample = 0
        self.main_receive_time = 0  

        self.dt = dt
        self.logging_on = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.first_arm = True
        self.armed = False
        self.nmpc_status = NMPCStatus.STARTING      
        self.count = 0 # number of times through main callback
        self.x_offset = 0.0  # offsets needed for optical flow
        self.y_offset = 0.0
        self.timer = self.create_timer(self.dt, self.main_loop)    


#############################  MAIN CALLBACK ####################################

    def main_loop(self):
        
        if self.nmpc_status == NMPCStatus.DISARM_REQUEST: 
            if self.armed:
                self.disarm()
            else:
                self.get_logger().info('MAIN Node shutting down')
                rclpy.shutdown()
        else: 
            if self.armed and self.nmpc_status == NMPCStatus.READY:
                self.publish_nmpc_input()

            self.count += 1
            self.maintain_offboard()

            # arming attempts are limited to one time per second
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
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)  # offboard mode
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0) # arm


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
            0.0,          # disarm
            21196.0       # FORCE (disarm in-air)
        )

    # send the latest state info to the nmpc node
    def publish_nmpc_input(self):
        msg = NMPCInput()
        msg.timestamp_sample = self.timestamp_sample
        msg.main_receive_time = self.main_receive_time
        msg.main_send_time = self.get_clock().now().nanoseconds // 1000
        msg.state = self.state
        msg.battery_voltage = self.voltage
        msg.filtered_voltage = self.filtered_voltage

        # additional info for thrust testing
        msg.current_a = self.current_a
        msg.current_average_a = self.current_average_a 
        msg.discharged_mah = self.discharged_mah 
        msg.remaining = self.remaining

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



    def battery_callback(self, msg):
        self.voltage = msg.voltage_v
        self.voltage_history.append(self.voltage)
        self.filtered_voltage = np.mean(self.voltage_history)
        self.current_a = msg.current_a
        self.current_average_a = msg.current_average_a
        self.discharged_mah = msg.discharged_mah
        self.remaining = msg.remaining




        

    def nmpc_status_callback(self, msg):
        self.nmpc_status = msg.status
        self.get_logger().info('NMPC status: ' + str(self.nmpc_status))


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