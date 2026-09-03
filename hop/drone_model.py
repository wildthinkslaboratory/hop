import numpy as np
import casadi as ca
from casadi import sin, cos
import do_mpc


class DroneModel:
    def __init__(self, mc):
        self.mc = mc

        self.model = do_mpc.model.Model('continuous' , 'SX')
        p = self.model.set_variable(var_type='_x', var_name='p', shape=(3,1))
        v = self.model.set_variable(var_type='_x', var_name='v', shape=(3,1))
        q = self.model.set_variable(var_type='_x', var_name='q', shape=(4,1))
        w = self.model.set_variable(var_type='_x', var_name='w', shape=(3,1))
        F = self.model.set_variable(var_type='_x', var_name='F', shape=(1,1))
        state = ca.vertcat(p,v,q,w,F)
        u = self.model.set_variable(var_type='_u', var_name='u', shape=(4,1))

        # Parameters 
        # -------------------
        # x position
        # y position
        # z position
        # battery voltage
        # goal thrust
        parameters = self.model.set_variable(var_type='_p', var_name='parameters', shape=(5,1))

        I_mat = ca.DM(mc.I)

        volt = parameters[3]
        F_ss = (
            mc.c0
            + mc.c1 * u[2]
            + mc.c2 * volt
            + mc.c3 * u[2]**2
            + mc.c4 * volt**2
            + mc.c5 * u[2]*volt
        )        

        M = mc.d * mc.Izz * u[3]

        theta_x = u[0] + mc.gimbal_offset[0]
        theta_y = u[1] + mc.gimbal_offset[1]

        F_vector = F * ca.vertcat(
            sin((np.pi/180) * theta_y),
            -sin((np.pi/180) * theta_x)*cos((np.pi/180) * theta_y),
            cos((np.pi/180) * theta_x)*cos((np.pi/180) * theta_y)
        )
        

        roll_moment = ca.vertcat(0, 0, M)
        M_vector = ca.cross(mc.moment_arm, F_vector) + roll_moment
        angular_momentum = I_mat @ w

        q_full = state[6:10]
        q_full = q_full / ca.norm_2(q_full)

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

        self.model.set_rhs('p', v)
        self.model.set_rhs('v', (r_b2w @ F_vector) / mc.m + mc.g)
        self.model.set_rhs('q', 0.5 * Q_omega @ q_full)
        self.model.set_rhs('w', ca.solve(I_mat, M_vector - ca.cross(w, angular_momentum)))
        self.model.set_rhs('F', (F_ss - F) / mc.tau)


        # build the cost function
        x_r = ca.vertcat(parameters[:3], mc.xr[3:])
        
        x_error = state - x_r
        x_cost = x_error.T @ mc.Q @ x_error 
        terminal_cost = x_error.T @ (mc.terminal_cost_factor * mc.Q) @ x_error 
        u_goal = ca.vertcat(mc.gimbal_offset[0], mc.gimbal_offset[1], mc.hover_thrust, 0.0)
        u_error = u - u_goal
        u_cost = u_error.T @ mc.R @ u_error
        cost = x_cost + u_cost
        self.model.set_expression(expr_name='terminal_cost', expr=terminal_cost )
        self.model.set_expression(expr_name='cost', expr=cost)

        self.model.setup()
