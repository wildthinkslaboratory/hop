import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import VehicleOdometry

from std_msgs.msg import Float64MultiArray

import numpy as np
import casadi as ca
from casadi import sin, cos
import do_mpc
from hop.constants import Constants

mc = Constants()

class NMPC(Node):

    def __init__(self):
        super().__init__('NMPC')

        self.publisher_ = self.create_publisher(Float64MultiArray, 'NMPC', 10)

        self.subscription = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.listener_callback,
            qos_profile_sensor_data)
        self.subscription

        self.dt = mc.dt
        self.state = mc.x0

        self.model = self.build_model()
        self.mpc = self.initialize_NMPC()

        self.timer = self.create_timer(self.dt, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        control = self.run_NMPC()
        pwm_output = self.control_translator(control)
        msg.data = pwm_output.flatten().tolist()
        self.publisher_.publish(msg)
        state_np = np.array(self.state).flatten()
        control_np = np.array(control).flatten()

        self.get_logger().info(
            f"""\n=== NMPC Step ===
        State:
        x: {state_np[0:3]}
        v: {state_np[3:6]}
        q: {state_np[6:9]}
        w: {state_np[9:12]}
        Control:
        u: {control_np}
        """
        )

    def listener_callback(self, msg):
        state = [0.0] * 12
        q_full = np.array(msg.q)
        norm = np.linalg.norm(q_full)
        if norm > 0:
            q_full /= norm
        state[0:3] = msg.position
        state[3:6] = msg.velocity
        state[6:9] = q_full[1:4]
        state[9:12] = msg.angular_velocity
        self.state = ca.DM(state)

    def build_model(self):
        model_type = 'continuous' 
        symvar_type = 'SX'
        model = do_mpc.model.Model(model_type, symvar_type)

        x = model.set_variable(var_type='_x', var_name='x', shape=(3,1))
        v = model.set_variable(var_type='_x', var_name='v', shape=(3,1))
        q = model.set_variable(var_type='_x', var_name='q', shape=(3,1))
        w = model.set_variable(var_type='_x', var_name='w', shape=(3,1))
    
        state = ca.vertcat(x,v,q,w)

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

        qw = (1 - (state[6])**2 - (state[7])**2 - (state[8])**2)**(0.5)

        r_b2w = ca.vertcat(
            ca.horzcat(1 - 2*(state[7]**2 + state[8]**2), 2*(state[6]*state[7] - state[8]*qw), 2*(state[6]*state[8] + state[7]*qw)),
            ca.horzcat(2*(state[6]*state[7] + state[8]*qw), 1 - 2*(state[6]**2 + state[8]**2), 2*(state[7]*state[8] - state[6]*qw)),
            ca.horzcat(2*(state[6]*state[8] - state[7]*qw), 2*(state[7]*state[8] + state[6]*qw), 1 - 2*(state[6]**2 + state[7]**2)),
        )

        Q_omega = ca.vertcat(
            ca.horzcat(0, state[11], -state[10], state[9]),
            ca.horzcat(-state[11], 0, state[9], state[10]),
            ca.horzcat(state[10], -state[9], 0, state[11]),
            ca.horzcat(state[9], state[10], -state[11], 0)
        )

        q_full = ca.vertcat(state[6:9], qw)

        q_full = q_full / ca.norm_2(q_full)

        model.set_rhs('x', v)
        model.set_rhs('v', (r_b2w @ F_vector) / mc.m + mc.g)
        model.set_rhs('q', 0.5 * Q_omega[0:3, :] @ q_full)
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
        x = ca.vertcat(self.model.x['x'], self.model.x['v'], self.model.x['q'], self.model.x['w'])
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

        du_bounds = mc.theta_dot_constraint * self.dt

        mpc.bounds['lower','_du','u'] = [-du_bounds, -du_bounds, -np.inf, -np.inf]
        mpc.bounds['upper','_du','u'] = [ du_bounds,  du_bounds,  np.inf,  np.inf]

        mpc.bounds['lower', '_x', 'p'] = [-ca.inf, -ca.inf, 0]
        mpc.bounds['upper', '_x', 'p'] = [ ca.inf,  ca.inf,  ca.inf]

        u = self.model.u['u']

        P_avg = u[2]
        P_diff = u[3]

        P_upper = P_avg + P_diff / 2
        P_lower = P_avg - P_diff / 2

        mpc.set_nl_cons('upper_pwm_limit', P_upper, ub=mc.prop_thrust_constraint[1])
        mpc.set_nl_cons('upper_pwm_floor', P_upper, lb=mc.prop_thrust_constraint[0])
        mpc.set_nl_cons('lower_pwm_limit', P_lower, ub=mc.prop_thrust_constraint[1])
        mpc.set_nl_cons('lower_pwm_floor', P_lower, lb=mc.prop_thrust_constraint[0])



    def initialize_NMPC(self):
        mpc = self.build_NMPC()
        mpc.x0 = self.state
        mpc.set_initial_guess()
        return mpc

    def run_NMPC(self):
        control = self.mpc.make_step(self.state)
        return control
    
    def get_angle_pwm(self, gimbal_angles):
        gimbal_angles[0] = np.clip(gimbal_angles[0], mc.outer_gimbal_range[0], mc.outer_gimbal_range[1])
        gimbal_angles[1] = np.clip(gimbal_angles[1],  mc.inner_gimbal_range[0], mc.inner_gimbal_range[1])

        outer_angle_pwm = (500 * (gimbal_angles[0] / 180)) + 1500
        inner_angle_pwm = (500 * (gimbal_angles[1] / 180)) + 1500
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
