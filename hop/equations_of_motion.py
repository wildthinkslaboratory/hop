import casadi as ca
from casadi import sin, cos
import numpy as np

from hop.constants import Constants

# Making the equations of motion a separate class to assure that all of our NLP's 
# use the same equations. If we change the model it happens in one place.
# The dompc implementation doesn't use this though so you need to make sure that 
# drone_model.py (used by dompc) and this match so we have one set of equations.
class Equations6DOF:
    def __init__(self, mc):
        self.mc = mc

        # First create our state variables and control variables
        p = ca.SX.sym('p', 3, 1)
        v = ca.SX.sym('v', 3, 1)
        q = ca.SX.sym('q', 4, 1)
        w = ca.SX.sym('w', 3, 1)

        self.x = ca.vertcat(p,v,q,w)
        self.u = ca.SX.sym('u', 4, 1)

        print(self.x)

        # Parameters 
        # -------------------
        # x position
        # y position
        # z position
        # battery voltage
        # goal thrust
        self.p = ca.SX.sym('parameters', 5)

        # Now we build up the equations of motion and create a function
        # for the system dynamics
        I_mat = ca.DM(mc.I)

        top = self.u[2] - self.u[3]/2
        bot = self.u[2] + self.u[3]/2
        volt = self.p[3]
        F = (
            mc.c0
            + mc.c1 * top
            + mc.c2 * bot
            + mc.c3 * volt
            + mc.c4 * top**2
            + mc.c5 * bot**2
            + mc.c6 * volt**2
            + mc.c7 * top * bot
            + mc.c8 * top * volt
            + mc.c9 * bot * volt
        ) 
        M = mc.d * mc.Izz * self.u[3]


        theta_x = self.u[0] + mc.gimbal_offset[0]
        theta_y = self.u[1] + mc.gimbal_offset[1]

        F_vector = F * ca.vertcat(
            sin((np.pi/180) * theta_y),
            -sin((np.pi/180) * theta_x)*cos((np.pi/180) * theta_y),
            cos((np.pi/180) * theta_x)*cos((np.pi/180) * theta_y)
        )


        roll_moment = ca.vertcat(0, 0, M)
        M_vector = ca.cross(mc.moment_arm, F_vector) + roll_moment
        angular_momentum = I_mat @ w


        r_b2w = ca.vertcat(
            ca.horzcat(1 - 2*(self.x[7]**2 + self.x[8]**2), 2*(self.x[6]*self.x[7] - self.x[8]*self.x[9]), 2*(self.x[6]*self.x[8] + self.x[7]*self.x[9])),
            ca.horzcat(2*(self.x[6]*self.x[7] + self.x[8]*self.x[9]), 1 - 2*(self.x[6]**2 + self.x[8]**2), 2*(self.x[7]*self.x[8] - self.x[6]*self.x[9])),
            ca.horzcat(2*(self.x[6]*self.x[8] - self.x[7]*self.x[9]), 2*(self.x[7]*self.x[8] + self.x[6]*self.x[9]), 1 - 2*(self.x[6]**2 + self.x[7]**2)),
        )

        Q_omega = ca.vertcat(
            ca.horzcat(0, self.x[12], -self.x[11], self.x[10]),
            ca.horzcat(-self.x[12], 0, self.x[10], self.x[11]),
            ca.horzcat(self.x[11], -self.x[10], 0, self.x[12]),
            ca.horzcat(-self.x[10], -self.x[11], -self.x[12], 0)
        )

        q_full = self.x[6:10]
        q_full = q_full / ca.norm_2(q_full)

        self.RHS = ca.vertcat(
            v,
            (r_b2w @ F_vector) / mc.m + mc.g,
            0.5 * Q_omega @ q_full,
            ca.solve(I_mat, M_vector - ca.cross(w, angular_momentum))
        )

        # f is function that returns the change in state for a given state and control values
        self.f = ca.Function('f', [self.x, self.u, self.p], [self.RHS])

        # this is helpful for thrust modeling
        self.T = ca.Function('T', [self.u, self.p], [F_vector])