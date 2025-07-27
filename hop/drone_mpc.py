import do_mpc
import numpy as np
import casadi as ca

# from hop.constants import Constants
from constants import Constants

mc = Constants()

class DroneMPC:
    def __init__(self, dt, model):
        self.dt = dt
        self.model = model
        self.mpc = do_mpc.controller.MPC(self.model)
        
        self.mpc.settings.n_horizon = 10
        self.mpc.settings.n_robust = 1
        self.mpc.settings.open_loop = 0
        self.mpc.settings.t_step = self.dt
        self.mpc.settings.state_discretization = 'collocation'
        self.mpc.settings.collocation_type = 'radau'
        self.mpc.settings.collocation_deg = 2
        self.mpc.settings.collocation_ni = 1
        self.mpc.settings.store_full_solution = True

        self.mpc.bounds['lower', '_u', 'u'] = [
            mc.outer_gimbal_range[0],
            mc.inner_gimbal_range[0],
            -np.inf,
            mc.diff_thrust_constraint[0]
        ]

        self.mpc.bounds['upper', '_u', 'u'] = [
            mc.outer_gimbal_range[1],
            mc.inner_gimbal_range[1],
            np.inf,
            mc.diff_thrust_constraint[1]
        ]

        # restrict rate of change for gimbals
        # du_bounds = [c * self.dt for c in mc.theta_dot_constraint]
        # self.mpc.bounds['lower','_du','u'] = [du_bounds[0], du_bounds[0], -np.inf, -np.inf]
        # self.mpc.bounds['upper','_du','u'] = [du_bounds[1], du_bounds[1],  np.inf,  np.inf]

        self.mpc.bounds['lower', '_x', 'p'] = [-ca.inf, -ca.inf, 0]
        self.mpc.bounds['upper', '_x', 'p'] = [ ca.inf,  ca.inf,  ca.inf]

        
        # u = self.model.u['u']
        # P_avg = u[2]
        # P_diff = u[3]
        # P_upper = P_avg + P_diff / 2
        # P_lower = P_avg - P_diff / 2

        # self.mpc.set_nl_cons('upper_pwm_limit', P_upper, ub=mc.prop_thrust_constraint[1])
        # self.mpc.set_nl_cons('upper_pwm_floor', P_upper, lb=mc.prop_thrust_constraint[0])
        # self.mpc.set_nl_cons('lower_pwm_limit', P_lower, ub=mc.prop_thrust_constraint[1])
        # self.mpc.set_nl_cons('lower_pwm_floor', P_lower, lb=mc.prop_thrust_constraint[0])

        self.mpc.settings.supress_ipopt_output()
        


    def set_start_state(self, x0):
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    def set_goal_state(self, x_r):
        Q = mc.Q
        x = ca.vertcat(self.model.x['p'], self.model.x['v'], self.model.x['q'], self.model.x['w'])
        error = x - x_r
        lterm = error.T @ Q @ error
        self.mpc.set_objective(lterm=lterm, mterm=lterm)
        self.mpc.set_rterm(u=np.array([1, 1, 1, 1], dtype=float))
        self.mpc.setup()
