from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import BatteryStatus, VehicleStatus, ActuatorMotors, ActuatorServos, OffboardControlMode, VehicleStatus, VehicleCommand, VehicleOdometry
from rclpy.qos import qos_profile_sensor_data
from time import perf_counter

from casadi import DM
import numpy as np
from scipy.spatial.transform import Rotation as R
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

class OffBoardNode(Node):

    def __init__(self, name, timelimit = None, dt = mc.dt):
        super().__init__(name)

        self.logging_on = False
        self.timelimit = timelimit
        self.dt = dt
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
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

        # state subscription      
        self.vehicle_odometry = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.state_callback,
            qos_profile_sensor_data
        )

        self.vehicle_status = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            qos_profile_sensor_data
        )

        self.battery_status = self.create_subscription(
            BatteryStatus,
            '/fmu/out/battery_status',
            self.battery_callback,
            qos_pub
        )


        # we create our own thread to listen for keyboard strokes
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.running = True
        self.listener_thread = threading.Thread(target=self.keyboard_callback, daemon=True)
        self.listener_thread.start()
        self.key = ''
        self.waypoint_i = 0


        ############ Publishers #########################

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

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
        self.state = mc.x0
        self.armed = False
        self.control = np.array([0.0, 0.0, 0.0, 0.0])
        self.pwm_motors = [0.0, 0.0]
        self.pwm_servos = [0.0, 0.0]
        self.log_rows = []
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.count = 0
        self.x_offset = 0.0  # offsets needed for optical flow
        self.y_offset = 0.0
        self.voltage = 0.0




############################# callbacks  ####################################

    # publish all of our messages
    def timer_callback(self):

        self.count += 1
        if not self.timelimit == None and self.count * self.dt > self.timelimit:
            raise SystemExit  # time to exit node

        self.maintain_offboard()
        self.run_motors()
        self.run_servos()

        if not self.armed and self.count >= 100 and self.count % 50 == 0:
            self.offboard_arm()

        self.log_rows.append({
            'state': self.state.full().flatten().tolist(),
            'control': self.control.tolist(),
            'pwm_motors': self.pwm_motors,
            'pwm_servos': self.pwm_servos,
            'voltage': self.voltage,
            'parameters': mc.waypoints[self.waypoint_i].tolist()
            # 'timestamp': perf_counter()
        })

    # recieve armed status
    def status_callback(self, msg):
        was_armed = self.armed
        if msg.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            self.armed = True
            self.get_logger().info('Vehicle is ARMED')
        elif msg.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
            self.armed = False
            self.get_logger().info('Vehicle is DISARMED')

        # if it switches from disarmed to armed then we set the x and y offsets
        if not was_armed and self.armed:
            self.x_offset = self.state[0]
            self.y_offset = self.state[1]


    # recieve armed status
    def battery_callback(self, msg):

        self.voltage = msg.voltage_v
        # self.get_logger().info('Voltage :' + str(self.voltage))


    # recieve vehicle odometry message
    def state_callback(self, msg):
        state = [0.0] * 13

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
        state[0:3] = [pos[1] - self.x_offset, pos[0] - self.y_offset, -pos[2]]
        state[3:6] = [vel[1], vel[0], -vel[2]]
    
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
        state[6:10] = np.array([q_FLU_ENU[1], q_FLU_ENU[2], q_FLU_ENU[3], q_FLU_ENU[0]])
               
        ang_vel = msg.angular_velocity
        state[10:13] = [ang_vel[1], ang_vel[0], -ang_vel[2]]

        self.state = DM(state)

        # self.get_logger().info('offsets ' + str(self.state[0:3]) + ' ' + str(self.x_offset) + ' ' + str(self.y_offset))

        if self.logging_on:
            self.get_logger().info(
                f"""\n=== NMPC Step ===
                State:
                p: {state[0:3]}
                v: {state[3:6]}
                q: {state[6:10]}
                w: {state[10:13]}
                """
            )

    # receive keyboard strokes
    def keyboard_callback(self):    
        while self.running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                self.key = sys.stdin.read(1)
                if self.logging_on:
                    self.get_logger().info(f"Key pressed: {self.key}")


    # when we exit do clean up and output the run data
    def destroy_node(self):
        # shutting down keyboard listenting
        self.running = False
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

        self.disarm()

        data = {'constants': mc.__dict__(), 'run_data': self.log_rows}
        data = {'run_data': self.log_rows}
        output_data(data, "src/hop/plotter_logs/current.json")
        formatted_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
        output_data(data, "src/hop/plotter_logs/" + formatted_date + "log.json")
        super().destroy_node()

################################### PUBLISHER functions #######################################


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



    def run_motors(self):
        motor_command = ActuatorMotors()
        t = self.get_clock().now().nanoseconds // 1000
        motor_command.timestamp_sample = t
        motor_command.timestamp = t
        motor_command.control = [self.pwm_motors[0], self.pwm_motors[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 4 motors + 8 unused
        

        self.publisher_motors.publish(motor_command)
        if self.logging_on:
            self.get_logger().info('Publishing motor pwm ' + str(self.pwm_motors))


    def run_servos(self):
        servo_command = ActuatorServos()
        t = self.get_clock().now().nanoseconds // 1000
        servo_command.timestamp_sample = t
        servo_command.timestamp = t
        servo_command.control = [-self.pwm_servos[0], -self.pwm_servos[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 4 motors + 4 unused

        self.publisher_servos.publish(servo_command)
        if self.logging_on:
            self.get_logger().info('Publishing servo pwm ' + str(self.pwm_servos))


    def offboard_arm(self):
        # switch to Offboard
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
        # arm
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        if self.logging_on:
            self.get_logger().info('Publishing arm and offboard mode commands')

    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            0.0,          # param1 = disarm
            21196.0       # param2 = FORCE (disarm in-air)
        )

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




