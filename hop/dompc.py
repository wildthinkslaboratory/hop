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
        self.mpc.settings.collocation_ni = 1
        self.mpc.settings.t_step = mc.finite_interval_size    
        self.mpc.settings.n_horizon = mc.number_intervals
        self.mpc.settings.collocation_deg = mc.collocation_degree  

        # lower bounds on control
        self.mpc.bounds['lower', '_u', 'u'] = [
            mc.outer_gimbal_range[0],
            mc.inner_gimbal_range[0],
            -np.inf,
            -np.inf
        ]

        # upper bounds on control
        self.mpc.bounds['upper', '_u', 'u'] = [
            mc.outer_gimbal_range[1],
            mc.inner_gimbal_range[1],
            np.inf,
            np.inf
        ]

        # upper and lower bounds on position (x,y,z)
        self.mpc.bounds['lower', '_x', 'p'] = [-ca.inf, -ca.inf, 0]
        self.mpc.bounds['upper', '_x', 'p'] = [ ca.inf,  ca.inf,  ca.inf]

        # set max limit on each thrust motor
        control = self.model.u['u']
        P_upper = control[2] + control[3] / 2
        P_lower = control[2] - control[3] / 2
        thrust_limit = mc.prop_thrust_constraint
        self.mpc.set_nl_cons('upper_pwm_max', P_upper, ub=thrust_limit)
        self.mpc.set_nl_cons('lower_pwm_max', P_lower, ub=thrust_limit)


    def set_start_state(self, x0):
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    # set a positional waypoint
    # this gets fed into the NMPC as parameters that are
    # in the goal state.
    def set_waypoint(self, p_goal):
        self.p_goal['_p'] = p_goal

    def setup_cost(self):

        # set up the (x,y,z) position as parameters
        # so we can adjust the goal state with different waypoints
        self.p_goal = self.mpc.get_p_template(1)
        self.p_goal['_p'] = np.array([0.0, 0.0, 0.0])
        def p_fun(t_now):
            return self.p_goal
        self.mpc.set_p_fun(p_fun)
        
        # build the cost function
        x = ca.vertcat(self.model.x['p'], self.model.x['v'], self.model.x['q'], self.model.x['w'])
        x_r = ca.vertcat(self.model.p['p_goal'], mc.xr[3:])
        x_error = x - x_r
        x_cost = x_error.T @ mc.Q @ x_error 
        u_goal = ca.DM([0.0, 0.0, mc.hover_thrust, 0.0])
        u_error = self.model.u['u'] - u_goal
        u_cost = u_error.T @ mc.R @ u_error
        cost = x_cost + u_cost
        self.mpc.set_objective(lterm=cost, mterm=x_cost)

        if mc.nmpc_rate_constraints:
            self.mpc.set_rterm(u=np.array([1, 1, 1, 1], dtype=float))

        self.mpc.setup()