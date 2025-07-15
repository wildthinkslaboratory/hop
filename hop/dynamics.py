import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

import do_mpc
from do_mpc import estimator
import casadi as ca
from casadi import sin, cos
from hop.constants import Constants

mc = Constants()

class Dynamics(Node):

    def __init__(self):
        super().__init__('dynamics')

        self.publisher_ = self.create_publisher(Float64MultiArray, 'dynamics', 10)

        self.subscription = self.create_subscription(
            Float64MultiArray,
            'NMPC',
            self.listener_callback,
            10)
        self.subscription
        self.model = self.build_model()
        self.control = None

        self.dt = mc.dt
        x0 = ca.vertcat(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

        self.simulator = self.create_simulator(self.dt)
        self.estimator = self.create_estimator(x0)

    def timer_callback(self):
        if self.control is None:
            return
        msg = Float64MultiArray()
        state = self.run_dynamics()
        msg.data = state.flatten().tolist()
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)

    def listener_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)
        self.control = ca.DM(msg.data)
        if not hasattr(self, 'timer'):
            self.get_logger().info("Starting simulation timer.")
            self.timer = self.create_timer(self.dt, self.timer_callback)

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
    
    def create_simulator(self, dt):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step = dt)
        simulator.setup()
        return simulator

    def create_estimator(self, x0):
        estimator = do_mpc.estimator.StateFeedback(self.model)
        estimator.x0 = x0
        return estimator

    def run_dynamics(self):
        measurement = self.simulator.make_step(self.control)
        new_state = self.estimator.make_step(measurement)
        return new_state

    



def main(args=None):
    print('dynamics node')
    rclpy.init(args=args)

    dynamics = Dynamics()

    rclpy.spin(dynamics)
    dynamics.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

