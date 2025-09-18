import do_mpc
import numpy as np
import casadi as ca

from hop.constants import Constants

mc = Constants()

class DroneMPC:
    def __init__(self, dt, model):
        self.dt = dt
        self.model = model
        self.mpc = do_mpc.controller.MPC(self.model)


        self.mpc.settings.nlpsol_opts = mc.ipopt_settings
        self.mpc.settings.t_step = self.dt    # the length of a finite element ?
        self.mpc.settings.n_horizon = mc.mpc_horizon 
        self.mpc.settings.collocation_deg = 2  # the degree of polynomial and number of quadrature points per time step
        self.mpc.settings.collocation_ni = 1


        self.mpc.bounds['lower', '_u', 'u'] = [
            mc.outer_gimbal_range[0],
            mc.inner_gimbal_range[0],
            -np.inf,
            -np.inf
        ]

        self.mpc.bounds['upper', '_u', 'u'] = [
            mc.outer_gimbal_range[1],
            mc.inner_gimbal_range[1],
            np.inf,
            np.inf
        ]

        self.mpc.bounds['lower', '_x', 'p'] = [-ca.inf, -ca.inf, 0]
        self.mpc.bounds['upper', '_x', 'p'] = [ ca.inf,  ca.inf,  ca.inf]


        # set max limit on each thrust motor
        control = self.model.u['u']
        P_upper = control[2] + control[3] / 2
        P_lower = control[2] - control[3] / 2
        thrust_limit = mc.prop_thrust_constraint
        self.mpc.set_nl_cons('upper_pwm_max', P_upper, ub=thrust_limit)
        self.mpc.set_nl_cons('lower_pwm_max', P_lower, ub=thrust_limit)

        self.mpc.settings.supress_ipopt_output()
        


    def set_start_state(self, x0):
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    def set_goal_state(self, x_r):

        x = ca.vertcat(self.model.x['p'], self.model.x['v'], self.model.x['q'], self.model.x['w'])
        x_error = x - x_r
        x_cost = x_error.T @ mc.Q @ x_error 
        u_goal = ca.DM([0.0, 0.0, mc.hover_thrust, 0.0])
        u_error = self.model.u['u'] - u_goal
        u_cost = u_error.T @ mc.R @ u_error
        cost = x_cost + u_cost
        self.mpc.set_objective(lterm=cost, mterm=x_cost)

        # this is rate change limitations for control
        # self.mpc.set_rterm(u=np.array([1, 1, 1, 1], dtype=float))


        self.mpc.prepare_nlp()
        self.mpc.create_nlp()

