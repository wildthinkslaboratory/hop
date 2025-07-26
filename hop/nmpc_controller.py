import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from std_msgs.msg import Float64MultiArray
from px4_msgs.msg import ActuatorMotors, ActuatorServos, OffboardControlMode, VehicleStatus, VehicleCommand, VehicleOdometry
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import casadi as ca
from casadi import sin, cos
import do_mpc
from hop.constants import Constants
from hop.plotter import DronePlotter3d

mc = Constants()

class NMPC(Node):

    def __init__(self):
        super().__init__('nmpc_controller')

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

        self.setpoint_cnt = 0

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

        self.dt = mc.dt
        self.state = mc.x0

        self.model = self.build_model()
        self.mpc = self.initialize_NMPC()

        self.log_rows = []

        self.timer = self.create_timer(self.dt, self.timer_callback)

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

    def run_motors(self, motor_pwm):
        motor_command = ActuatorMotors()
        t = self.get_clock().now().nanoseconds // 1000
        motor_command.timestamp_sample = t
        motor_command.timestamp = t
        motor_command.control = [motor_pwm[0], motor_pwm[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 4 motors + 8 unused
        self.publisher_motors.publish(motor_command)
        # self.get_logger().info('Sent PWM vector: ' + str(msg.control[:4]))

    def run_servos(self, servo_pwm):
        servo_command = ActuatorServos()
        t = self.get_clock().now().nanoseconds // 1000
        servo_command.timestamp_sample = t
        servo_command.timestamp = t
        self.get_logger().info(
            f"""\n=== NMPC Step ===
            Control:
            pwm: {servo_pwm[0]}
            """
        )
        servo_command.control = [servo_pwm[0], servo_pwm[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 4 motors + 4 unused
        self.publisher_servos.publish(servo_command)
        # self.get_logger().info('Sent PWM sservo: ' + str(msg2.control[:5]))

    def offboard_arm(self):
        if self.setpoint_cnt == 10:
            # switch to Offboard
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
            # arm
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.setpoint_cnt += 1
    
    def maintain_offboard(self):
        # self.get_logger().info('send offboard')
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.direct_actuator = True
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.publisher_offboard_mode.publish(offboard_msg)

    def timer_callback(self):
        self.maintain_offboard()
        control = self.run_NMPC()
        pwm_output = self.control_translator(control)
        servo_pwm = pwm_output[0:2]
        motor_pwm = pwm_output[2:4]
        self.run_motors(motor_pwm)
        self.run_servos(servo_pwm)
        self.offboard_arm()
        x_np = np.array(self.state).flatten()
        self.log_rows.append(np.hstack([x_np, control]))
        # self.get_logger().info(
        #     f"""\n=== NMPC Step ===
        #     Control:
        #     u: {control}
        #     pwm: {servo_pwm[0]}
        #     """
        # )


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
        self.state = ca.DM(state)
        self.get_logger().info(
            f"""\n=== NMPC Step ===
            State:
            p: {state[0:3]}
            v: {state[3:6]}
            q: {state[6:10]}
            w: {state[10:13]}
            """
        )

    def build_model(self):
        model_type = 'continuous' 
        symvar_type = 'SX'
        model = do_mpc.model.Model(model_type, symvar_type)

        p = model.set_variable(var_type='_x', var_name='p', shape=(3,1))
        v = model.set_variable(var_type='_x', var_name='v', shape=(3,1))
        q = model.set_variable(var_type='_x', var_name='q', shape=(4,1))
        w = model.set_variable(var_type='_x', var_name='w', shape=(3,1))
    
        state = ca.vertcat(p,v,q,w)

        u = model.set_variable(var_type='_u', var_name='u', shape=(4,1))

        I_mat = ca.diag(mc.I_diag)

        F = mc.a * u[2]**2 + mc.b * u[2] + mc.c
        M = mc.d * mc.Izz * u[3]

        F_vector = F * ca.vertcat(
            sin(u[1]),
            -sin(u[0])*cos(u[1]),
            cos(u[0])*cos(u[1])
        )

        roll_moment = ca.vertcat(0, 0, M)
        M_vector = ca.cross(mc.moment_arm, F_vector) + roll_moment

        angular_momentum = I_mat @ w

        r_b2w = ca.vertcat(
            ca.horzcat(1 - 2*(state[7]**2 + state[8]**2), 2*(state[6]*state[7] - state[8]*state[9]), 2*(state[6]*state[8] + state[7]*state[9])),
            ca.horzcat(2*(state[6]*state[7] + state[8]*state[9]), 1 - 2*(state[6]**2 + state[8]**2), 2*(state[7]*state[8] - state[6]*state[9])),
            ca.horzcat(2*(state[6]*state[8] - state[7]*state[9]), 2*(state[7]*state[8] + state[6]*state[9]), 1 - 2*(state[6]**2 + state[7]**2)),
        )

        Q_omega = ca.vertcat(
            ca.horzcat(0, state[12], -state[11], state[10]),
            ca.horzcat(-state[12], 0, state[10], state[11]),
            ca.horzcat(state[11], -state[10], 0, state[12]),
            ca.horzcat(-state[10], -state[11], -state[12], 0)
        )

        q_full = state[6:10]

        q_full = q_full / ca.norm_2(q_full)

        model.set_rhs('p', v)
        model.set_rhs('v', (r_b2w @ F_vector) / mc.m + mc.g)
        model.set_rhs('q', 0.5 * Q_omega @ q_full)
        model.set_rhs('w', ca.solve(I_mat, M_vector - ca.cross(w, angular_momentum)))

        model.setup()
        
        return model
    
    def build_NMPC(self):
        silence_solver=True
        mpc = do_mpc.controller.MPC(self.model)
        
        mpc.settings.n_horizon = 10
        mpc.settings.n_robust = 1
        mpc.settings.open_loop = 0
        mpc.settings.t_step = self.dt
        mpc.settings.state_discretization = 'collocation'
        mpc.settings.collocation_type = 'radau'
        mpc.settings.collocation_deg = 2
        mpc.settings.collocation_ni = 1
        mpc.settings.store_full_solution = True

        self.set_constraints(mpc)

        if silence_solver:
            mpc.settings.supress_ipopt_output()
        
        Q = mc.Q
        xr = mc.xr
        x = ca.vertcat(self.model.x['p'], self.model.x['v'], self.model.x['q'], self.model.x['w'])
        error = x - xr

        lterm = error.T @ Q @ error

        mpc.set_objective(lterm=lterm, mterm=lterm)

        mpc.set_rterm(u=np.array([1, 1, 1, 1], dtype=float))
        mpc.setup()

        return mpc
    
    def set_constraints(self, mpc):
        mpc.bounds['lower', '_u', 'u'] = [
            mc.outer_gimbal_range[0],
            mc.inner_gimbal_range[0],
            -np.inf,
            mc.diff_thrust_constraint[0]
        ]

        mpc.bounds['upper', '_u', 'u'] = [
            mc.outer_gimbal_range[1],
            mc.inner_gimbal_range[1],
            np.inf,
            mc.diff_thrust_constraint[1]
        ]

        # du = mpc.model._u - mpc.model._u_prev
        # mpc.set_nl_cons('du_rate_limit', du, lb=-du_bounds, ub=du_bounds)

        mpc.bounds['lower', '_x', 'p'] = [-ca.inf, -ca.inf, 0]
        mpc.bounds['upper', '_x', 'p'] = [ ca.inf,  ca.inf,  ca.inf]


        # P_avg = self.model.u[2]
        # P_diff = self.model.u[3]

        # P_upper = P_avg + P_diff / 2
        # P_lower = P_avg - P_diff / 2

        # pmin, pmax = mc.prop_thrust_constraint

        # self.mpc.set_nl_cons('upper_pwm_max', P_upper, ub=pmax)
        # self.mpc.set_nl_cons('upper_pwm_min', P_upper, lb=pmin)
        # self.mpc.set_nl_cons('lower_pwm_max', P_lower, ub=pmax)
        # self.mpc.set_nl_cons('lower_pwm_min', P_lower, lb=pmin)

    def initialize_NMPC(self):
        mpc = self.build_NMPC()
        mpc.x0 = self.state
        mpc.set_initial_guess()
        return mpc

    def run_NMPC(self):
        control = self.mpc.make_step(self.state)
        control = np.array(control).flatten()
        return control
    
    def get_angle_pwm(self, gimbal_angles):
        gimbal_angles[0] = np.clip(gimbal_angles[0], mc.outer_gimbal_range[0], mc.outer_gimbal_range[1])
        gimbal_angles[1] = np.clip(gimbal_angles[1],  mc.inner_gimbal_range[0], mc.inner_gimbal_range[1])

        outer_angle_pwm = gimbal_angles[0] / 90
        inner_angle_pwm = gimbal_angles[1] / 90
        return outer_angle_pwm, inner_angle_pwm
    
    def get_thrust_pwm(self, thrust_values):
        top_prop_pwm = thrust_values[0] + thrust_values[1]/2
        bottom_prop_pwm = thrust_values[0] - thrust_values[1]/2
        top_prop_pwm = np.clip(top_prop_pwm, mc.prop_thrust_constraint[0], mc.prop_thrust_constraint[1])
        bottom_prop_pwm = np.clip(bottom_prop_pwm, mc.prop_thrust_constraint[0], mc.prop_thrust_constraint[1])
        return top_prop_pwm, bottom_prop_pwm

    def control_translator(self, control):
        gimbal_angles = control[0:2]
        thrust_values = control[2:4]
        outer_angle, inner_angle = self.get_angle_pwm(gimbal_angles)
        top_prop_pwm, bottom_prop_pwm = self.get_thrust_pwm(thrust_values)
        pwm = [outer_angle, inner_angle, top_prop_pwm, bottom_prop_pwm]
        return pwm

    def finalize(self):
        if not self.log_rows:
            return
        data = np.vstack(self.log_rows)
        np.savez('run.npz', data=data, dt=self.dt)
        t = data.shape[0] * self.dt
        sim = DronePlotter3d(data, self.dt, t)
        sim.plot()

def main(args=None):
    rclpy.init(args=args)
    nmpc = NMPC()
    try:
        rclpy.spin(nmpc)
    except KeyboardInterrupt:
        pass
    finally:
        nmpc.finalize()
        nmpc.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

