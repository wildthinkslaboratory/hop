import rclpy
from rclpy.node import Node

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
            Float64MultiArray,
            'dynamics',
            self.listener_callback,
            10)
        self.subscription

        self.dt = mc.dt
        self.state = ca.vertcat(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)


        self.model = self.build_model()
        self.mpc = self.initialize_NMPC()

        self.timer = self.create_timer(self.dt, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        control = self.run_NMPC()
        msg.data = control.flatten().tolist()
        self.publisher_.publish(msg)
        state_np = np.array(self.state).flatten()
        control_np = np.array(control).flatten()

        # self.get_logger().info(
        #     f"""\n=== NMPC Step ===
        # State:
        # x: {state_np[0:3]}
        # v: {state_np[3:6]}
        # q: {state_np[6:9]}
        # w: {state_np[9:12]}
        # Control:
        # u: {control_np}
        # """
        # )

    def listener_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)
        self.state = ca.DM(msg.data)

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

        F = u[2] * ca.vertcat(
            sin(u[1]),
            -sin(u[0])*cos(u[1]),
            cos(u[0])*cos(u[1])
        )

        roll_moment = ca.vertcat(0, 0, u[3])
        M = ca.cross(mc.moment_arm, F) + roll_moment

        angular_momentum = ca.diag(mc.I_diag) @ w

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

        model.set_rhs('x', v)
        model.set_rhs('v', (r_b2w @ F) / mc.m + mc.g)
        model.set_rhs('q', 0.5 * Q_omega[0:3, :] @ q_full)
        model.set_rhs('w', ca.solve(ca.diag(mc.I_diag), M - ca.cross(w, angular_momentum)))

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

        if silence_solver:
            mpc.settings.supress_ipopt_output()
        
        Q = ca.diag(12)
        xr = ca.vertcat(1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        x = ca.vertcat(self.model.x['x'], self.model.x['v'], self.model.x['q'], self.model.x['w'])
        error = x - xr

        lterm = error.T @ Q @ error

        mpc.set_objective(lterm=lterm, mterm=lterm)

        mpc.set_rterm(u=np.array([1, 1, 1, 1], dtype=float))
        mpc.setup()
        return mpc

    def initialize_NMPC(self):
        mpc = self.build_NMPC()
        mpc.x0 = self.state
        mpc.set_initial_guess()
        return mpc

    def run_NMPC(self):
        control = self.mpc.make_step(self.state)
        return control




    



def main(args=None):
    rclpy.init(args=args)

    nmpc = NMPC()

    rclpy.spin(nmpc)
    nmpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

