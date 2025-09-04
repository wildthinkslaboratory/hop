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
        
        self.mpc.settings.n_horizon = mc.mpc_horizon
        self.mpc.settings.n_robust = 1
        self.mpc.settings.open_loop = 0
        self.mpc.settings.t_step = self.dt
        self.mpc.settings.state_discretization = 'collocation'
        self.mpc.settings.collocation_type = 'radau'
        self.mpc.settings.collocation_deg = 2
        self.mpc.settings.collocation_ni = 1
        self.mpc.settings.store_full_solution = True
        self.mpc.settings.nlpsol_opts = mc.ipopt_settings

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
        lterm = x_error.T @ mc.Q @ x_error 
        self.mpc.set_objective(lterm=lterm, mterm=lterm)
        self.mpc.set_rterm(u=np.array([1, 1, 1, 1], dtype=float))


        self.mpc.prepare_nlp()

        # # this creates bounds on the rate of change of the servos
        # ulist = self.mpc.opt_x['_u']
        # for i in range(len(ulist)):
        #     if not i == 0:
        #         rate_constraint = self.mpc.opt_x['_u', i, 0][:2] - self.mpc.opt_x['_u', i-1, 0][:2]
        #         self.mpc.nlp_cons.append(rate_constraint)
        #         shape = rate_constraint.shape
        #         self.mpc.nlp_cons_lb.append(-np.array([mc.theta_dot_constraint]*shape[0]).reshape(shape))
        #         self.mpc.nlp_cons_ub.append(np.array([mc.theta_dot_constraint]*shape[0]).reshape(shape))


        self.mpc.create_nlp()

