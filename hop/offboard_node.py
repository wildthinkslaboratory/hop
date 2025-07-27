from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from px4_msgs.msg import ActuatorMotors, ActuatorServos, OffboardControlMode, VehicleStatus, VehicleCommand, VehicleOdometry
from rclpy.qos import qos_profile_sensor_data

import numpy as np
from hop.constants import Constants
from casadi import DM
from hop.utilities import output_data
from datetime import datetime
mc = Constants()


class OffBoardNode(Node):

    def __init__(self, name, timelimit = None, dt = mc.dt):
        super().__init__(name)

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
        
        self.vehicle_odometry = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.listener_callback,
            qos_profile_sensor_data
        )

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

        self.state = mc.x0
        self.control = [0.0, 0.0, 0.0, 0.0]
        self.pwm_motors = [0.0, 0.0]
        self.pwm_servos = [0.0, 0.0]
        self.log_rows = []
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.count = 0
        self.logging_on = True

#############################   ####################################

    # publish all of our messages
    def timer_callback(self):

        self.count += 1
        if not self.timelimit == None and self.count * self.dt > self.timelimit:
            raise SystemExit  # time to exit node

        if self.count > 10:
            self.offboard_arm()

        self.maintain_offboard()
        self.run_motors()
        self.run_servos()

        self.log_rows.append({
            'state': self.state.full().tolist(),
            'control': self.control,
            'pwm_motors': self.pwm_motors,
            'pwm_servos': self.pwm_servos
        })

    # recieve vehicle odometry message
    def listener_callback(self, msg):
        state = [0.0] * 13
        q_full = np.array(msg.q)
        norm = np.linalg.norm(q_full)
        if norm > 0:
            q_full /= norm
        state[0:3] = msg.position
        state[3:6] = msg.velocity
        state[6:10] = [q_full[1], q_full[2], q_full[3], q_full[0]]
        state[10:13] = msg.angular_velocity
        self.state = DM(state)
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


    def finalize(self):
        data = {'constants': mc, 'run_data': self.log_rows}
        output_data(data, "src/hop/plotter_logs/current.json")
        formatted_date = datetime.now().strftime("%Y-%m-%d")
        output_data(data, "src/hop/plotter_logs/" + formatted_date + "log.json")


################################### PUBLISHER functions #######################################


    def publish_vehicle_command(self, command, p1=0., p2=0.):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.command = command
        msg.param1, msg.param2 = float(p1), float(p2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
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
        servo_command.control = [self.pwm_servos[0], self.pwm_servos[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 4 motors + 4 unused
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




