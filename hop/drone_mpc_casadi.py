import casadi as ca
from casadi import sin, cos
import numpy as np

from hop.constants import Constants
mc = Constants()

class DroneNMPCCasadi:
    def __init__(self):

        self.N = mc.mpc_horizon

        # First create our state variables and control variables
        p = ca.SX.sym('p', 3, 1)
        v = ca.SX.sym('v', 3, 1)
        q = ca.SX.sym('q', 4, 1)
        w = ca.SX.sym('w', 3, 1)
        self.x = ca.vertcat(p,v,q,w)
        self.u = ca.SX.sym('u', 4, 1)

        # Now we build up the equations of motion and create a function
        # for the system dynamics
        I_mat = ca.diag(mc.I_diag)
        F = mc.a * self.u[2]**2 + mc.b * self.u[2] + mc.c 
        M = mc.d * mc.Izz * self.u[3]

        F_vector = F * ca.vertcat(
            sin((np.pi/180)*self.u[1]),
            -sin((np.pi/180)*self.u[0])*cos((np.pi/180)*self.u[1]),
            cos((np.pi/180)*self.u[0])*cos((np.pi/180)*self.u[1])
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

        RHS = ca.vertcat(
            v,
            (r_b2w @ F_vector) / mc.m + mc.g,
            0.5 * Q_omega @ q_full,
            ca.solve(I_mat, M_vector - ca.cross(w, angular_momentum))
        )

        # f is function that returns the change in state for a given state and control values
        self.f = ca.Function('f', [self.x, self.u], [RHS])
        

     # In this function we build up the NMPC problem instance
    def set_goal_state(self, x_goal):

        self.x_goal = x_goal

        Q = ca.DM.eye(13)           # weights for cost of state errors
        R = ca.DM.eye(4) * 0.01     # weights for cost of motors
        X0 = ca.SX.sym('X0', self.x.size1())            # these are variables representing our initial state

        # we make a copy of the state variables for each N+1 time steps
        X = ca.SX.sym('X', self.x.size1(), self.N+1)    

        # we make a copy of the control variables for each N time steps
        U = ca.SX.sym('U', self.size_u(), self.N)       

        # here we build up the cost function by summing up the squared
        # error from the goal state over each time step
        cost = 0.0
        for k in range(self.N):
            state_k = X[:, k]    # state at time step k
            control_k = U[:, k]  # control at time step k

            # build up the cost function
            state_error_cost = (state_k - self.x_goal).T @ Q @ (state_k - self.x_goal)
            control_cost = control_k.T @ R @ control_k
            cost = cost + state_error_cost + control_cost

        # g will hold the 'equality' constraints. 
        # the right hand side is required to equal 0
        # this is the initial value constraint
        g = X[:, 0] - X0  

        for k in range(self.N):
            state_k = X[:, k]    # state at time step k
            control_k = U[:, k]  # control at time step k

            # now we build up the discrete constraints for the state
            # with the runge kutta method
            next_state = X[:, k+1]
            k1 = self.f(state_k, control_k)
            k2 = self.f(state_k + mc.dt/2*k1, control_k)
            k3 = self.f(state_k + mc.dt/2*k2, control_k)
            k4 = self.f(state_k + mc.dt * k3, control_k)
            next_state_RK4 = state_k + (mc.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, next_state - next_state_RK4)

        # We could add a terminal cost here, but we won't just yet
# -------------------------------------------------------------------------------------------------------
        # We make one long list of all the optimization variables
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        num_vars = opt_vars.numel()

        # now we add upper and lower bounds on the optimization variables
        # start with just negative infinity to positive infinity for everything
        self.lbx = -np.inf * np.ones(num_vars)
        self.ubx =  np.inf * np.ones(num_vars)

        # we have specific lower and upper bounds for the gimbals and delta thrust
        self.lbx[self.size_x()*(self.N+1):   num_vars: 4] = mc.outer_gimbal_range[0]     # outer gimbal lower bound
        self.lbx[self.size_x()*(self.N+1)+1: num_vars: 4] = mc.inner_gimbal_range[0]     # inner gimbal lower bound
        self.lbx[self.size_x()*(self.N+1)+3: num_vars: 4] = mc.diff_thrust_constraint[0] # delta thrust lower bound

        self.ubx[self.size_x()*(self.N+1):   num_vars: 4] = mc.outer_gimbal_range[1]     # outer gimbal upper bound
        self.ubx[self.size_x()*(self.N+1)+1: num_vars: 4] = mc.inner_gimbal_range[1]     # inner gimbal upper bound
        self.ubx[self.size_x()*(self.N+1)+3: num_vars: 4] = mc.diff_thrust_constraint[1] # delta thrust upper bound


            # limit on thrust motors
            # will add later
                    # # set max limit on each thrust motor
                    # control = self.model.u['u']
                    # P_upper = control[2] + control[3] / 2
                    # P_lower = control[2] + control[3] / 2
                    # thrust_limit = mc.prop_thrust_constraint
                    # self.mpc.set_nl_cons('upper_pwm_max', P_upper, ub=thrust_limit)
                    # self.mpc.set_nl_cons('lower_pwm_max', P_lower, ub=thrust_limit)
                

            # bounds on gimbal rate of change
            # will add later
                    # # this creates bounds on the rate of change of the servos
                    # ulist = self.mpc.opt_x['_u']
                    # for i in range(len(ulist)):
                    #     if not i == 0:
                    #         rate_constraint = self.mpc.opt_x['_u', i, 0][:2] - self.mpc.opt_x['_u', i-1, 0][:2]
                    #         self.mpc.nlp_cons.append(rate_constraint)
                    #         shape = rate_constraint.shape
                    #         self.mpc.nlp_cons_lb.append(-np.array([mc.theta_dot_constraint]*shape[0]).reshape(shape))
                    #         self.mpc.nlp_cons_ub.append(np.array([mc.theta_dot_constraint]*shape[0]).reshape(shape))

# -----------------------------------------------------------------------------------------------------------------------
        # Now we set up the solver and do all of the options and parameters

        # Define bounds for the constraints (all equality constraints are set to 0)
        g_bound = np.zeros(int(g.numel()))
        self.lbg = g_bound
        self.ubg = g_bound

        # dictionary for defining our solver
        nlp_prob = {
            'f': cost,
            'x': opt_vars,
            'g': g,
            'p': X0
        }

        # dictionary for our solver options
        opts = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def make_step(self, init_guess, x):
        return self.solver(x0=init_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=x)

    def set_start_state(self, x0):
        self.x0 = x0

    def size_u(self):
        return self.u.size1()
    
    def size_x(self):
        return self.x.size1()

        
solver = DroneNMPCCasadi()

sim_time = 10.0               # total simulation time (seconds)
n_sim_steps = int(sim_time/mc.dt)
tspan = np.arange(0, n_sim_steps * mc.dt, mc.dt)

# The starting state
x_current = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0]) 
x_initial_guess = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# Initial guess for the decision variables
# repeat x_0 across the horizon
X_init = np.tile(np.array(x_initial_guess).reshape(-1,1), (1, solver.N+1))
U_init = np.zeros((solver.size_u(), solver.N))

init_guess = np.concatenate([X_init.reshape(-1, order='F'),
                             U_init.reshape(-1, order='F')])

state_data = np.empty([n_sim_steps,13])
control_data = np.empty([n_sim_steps,4])

state_goal = ca.DM([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
solver.set_goal_state(state_goal)

for i in range(n_sim_steps):

    # Solve the NMPC for the current state x_current
    sol = solver.make_step(init_guess, x_current)

    sol_opt = sol['x'].full().flatten()
    
    # Extract the control sequence from the solution. The state trajectory is first,
    # so the first control is located at index = n_states*(N+1)
    u_current = sol_opt[solver.size_x() *(solver.N+1): solver.size_x()*(solver.N+1)+solver.size_u()]
    
    # Propagate the system using the discrete dynamics f (Euler forward integration)
    x_current = x_current + mc.dt* solver.f(x_current,u_current)
    
    state_data[i] = np.reshape(x_current, (13,))
    control_data[i] = np.reshape(u_current, (4,))

    


import sys
sys.path.append('..')
from plots import plot_state, plot_control
# first we print out the state in one plot
plot_state(tspan, state_data)
plot_control(tspan, control_data)
from animation import RocketAnimation
rc = RocketAnimation([-1, -0.1, -0.2], [0,1,0], 0.4)
rc.animate(tspan, state_data, control_data)
    