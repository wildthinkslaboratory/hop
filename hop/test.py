import rclpy
from rclpy.node import Node
from px4_msgs.msg import ActuatorMotors, ActuatorServos, OffboardControlMode, VehicleStatus, VehicleCommand
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import time

class SimplePwmTest(Node):
    def __init__(self):
        super().__init__('simple_pwm_test')
        
        # QoS profiles
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

        self.setpoint_cnt = 0

        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_sub)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.publisher_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_pub)

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_pub)        
        self.publisher_motors = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_pub)
        self.publisher_servos = self.create_publisher(ActuatorServos, '/fmu/in/actuator_servos', qos_pub)       
        self.timer = self.create_timer(0.1, self.send_pwm)  # 10 Hz


    def vehicle_status_callback(self, msg):
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state


    def publish_vehicle_command(self, command, p1=0., p2=0.):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.command           = command
        msg.param1, msg.param2 = float(p1), float(p2)
        msg.target_system     = 1
        msg.target_component  = 1
        msg.source_system     = 1
        msg.source_component  = 1
        msg.from_external     = True
        self.publisher_vehicle_command.publish(msg)


    def send_pwm(self):
        self.get_logger().info('send offboard')
        # publish off board control 
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.direct_actuator=True
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.publisher_offboard_mode.publish(offboard_msg)

        # If we are armed and offboard publish pwm for motors
        # if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
        msg = ActuatorMotors()
        t = self.get_clock().now().nanoseconds // 1000
        msg.timestamp_sample = t
        msg.timestamp = t
        msg.control = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 4 motors + 4 unused
        self.publisher_motors.publish(msg)
        self.get_logger().info('Sent PWM vector: ' + str(msg.control[:4]))

        msg2 = ActuatorServos()
        t = self.get_clock().now().nanoseconds // 1000
        msg2.timestamp_sample = t
        msg2.timestamp = t
        msg2.control = [0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0]   # 4 motors + 4 unused
        self.publisher_servos.publish(msg2)
        self.get_logger().info('Sent PWM sservo: ' + str(msg2.control[:5]))


        # Send mode‑change / arm once after 10 set‑points
        if self.setpoint_cnt == 10:
            # switch to Offboard
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
            # arm
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.setpoint_cnt += 1


def main(args=None):
    rclpy.init(args=args)
    node = SimplePwmTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()