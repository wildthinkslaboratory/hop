#   TODO:
# - there should be constraint that z > 0? Or z > ground?
# - starting gimbal angles of 0

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
     # we can't build it until we know the goal state
    def set_goal_state(self, x_goal):

        self.x_goal = x_goal

        Q = ca.DM.eye(13)           # weights for cost of state errors
        R = ca.DM.eye(4) * 0.01     # weights for cost of motors
        X0 = ca.SX.sym('X0', self.x.size1())            # these are variables representing our initial state

        # we make a copy of the state variables for each N+1 time steps
        X = ca.SX.sym('X', self.x.size1(), self.N+1)    

        # we make a copy of the control variables for each N time steps
        U = ca.SX.sym('U', self.size_u(), self.N)       

        # We make one long list of all the optimization variables
        # all the state variables preceed all the control variables.
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


        # g constraints contain an expression that is constrained by an upper and lower bound
        self.lbg = []   # will hold lower bounds for g constraints
        self.ubg = []   # will hold upper bounds for g constraints

        g = X[:, 0] - X0  
        self.lbg += [0.0]*int(g.numel())
        self.ubg += [0.0]*int(g.numel())


        cost = 0.0
        for k in range(self.N):
            x_k = X[:, k]    # state at time step k
            u_k = U[:, k]  # control at time step k

            # here we build up the cost function by summing up the squared
            # error from the goal state over each time step
            state_error_cost = (x_k - self.x_goal).T @ Q @ (x_k - self.x_goal)
            control_cost = u_k.T @ R @ u_k
            cost = cost + state_error_cost + control_cost

            # here we create the constraints that require the solution
            # to obey our system dynamics. We use Runge Kutta integration
            # and for each time step, we create a constraint that requires
            # the state at time k+1 to equal the system dynamics applied to the 
            # the state at time k.
            next_state = X[:, k+1]
            k1 = self.f(x_k, u_k)
            k2 = self.f(x_k + mc.dt/2*k1, u_k)
            k3 = self.f(x_k + mc.dt/2*k2, u_k)
            k4 = self.f(x_k + mc.dt * k3, u_k)
            next_state_RK4 = x_k + (mc.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, next_state - next_state_RK4)
            self.lbg += [0.0]*int(next_state.numel())
            self.ubg += [0.0]*int(next_state.numel())

            # build up the upper thrust limit constraints         
            g   = ca.vertcat(g, u_k[2] + 0.5*u_k[3] - mc.prop_thrust_constraint)
            g   = ca.vertcat(g, u_k[2] - 0.5*u_k[3] - mc.prop_thrust_constraint)
            self.lbg += [-ca.inf]*2
            self.ubg += [0.0]*2

            # build up rate of change constraints for servos 
            if k < self.N-1:
                next_u = U[:, k+1]  
                g   = ca.vertcat(g, u_k[0] - next_u[0] - mc.theta_dot_constraint)
                g   = ca.vertcat(g, u_k[1] - next_u[1] - mc.theta_dot_constraint)
                self.lbg += [-ca.inf]*2
                self.ubg += [0.0]*2



        # Now we set up the solver and do all of the options and parameters

        # dictionary for defining our solver
        nlp_prob = {
            'f': cost,
            'x': opt_vars,
            'g': g,
            'p': X0
        }

        # dictionary for our solver options
        opts = {
            'ipopt.max_iter': 200,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        
        
        # We create our initial guess for a solution.
        # Later we'll use the previous solution as our new solution guess.
        # repeat x_initial_guess across the horizon
        x_initial_guess = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        X_init = np.tile(np.array(x_initial_guess).reshape(-1,1), (1, self.N+1))
        # initial guess for controls is zero
        U_init = np.zeros((self.size_u(), self.N))

        # glue this all together to make our initial guess
        self.init_guess = np.concatenate([X_init.reshape(-1, order='F'),
                                    U_init.reshape(-1, order='F')])
        
        # Here we initialize our stored solution to zeros
        self.sol_x = np.zeros(self.size_x() * (self.N+1))
        self.sol_u = np.zeros(self.size_u() * self.N)
        self.first_iteration = True


    def make_step(self, x):

        # if it's not the first iteration, use a warm start from previous solution.
        # we shift the trajectory forward by on time step and then just repeat
        # the last timestep twice
        if self.first_iteration:
            self.first_iteration = False
        else:
            x_traj = np.concatenate([self.sol_x[self.size_x():], self.sol_x[self.size_x() * self.N:]])
            u_traj = np.concatenate([self.sol_u[self.size_u():], self.sol_u[self.size_u() * (self.N -1):]])
            self.init_guess = np.concatenate([x_traj, u_traj])

        # Call the NMPC solver 
        sol = self.solver(x0=self.init_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=x)
        sol_opt = sol['x'].full().flatten()

        # save the solution for warm starts
        self.sol_x = sol_opt[:self.size_x() *(self.N+1)]
        self.sol_u = sol_opt[self.size_x() *(self.N+1):]

        return self.sol_u[:self.size_u()] # return the first control step



    def set_start_state(self, x0):
        self.x0 = x0


    def size_u(self):
        return self.u.size1()
    
    def size_x(self):
        return self.x.size1()



# solver = DroneNMPCCasadi()

# sim_time = 10.0               # total simulation time (seconds)
# n_sim_steps = int(sim_time/mc.dt)
# tspan = np.arange(0, n_sim_steps * mc.dt, mc.dt)

# # The starting state
# x_current = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0]) 

# state_data = np.empty([n_sim_steps,13])
# control_data = np.empty([n_sim_steps,4])

# state_goal = ca.DM([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
# solver.set_goal_state(state_goal)

# for i in range(n_sim_steps):

#     # Solve the NMPC for the current state x_current
#     u_current = solver.make_step(x_current)
    
#     # Propagate the system using the discrete dynamics f (Euler forward integration)
#     x_current = x_current + mc.dt* solver.f(x_current,u_current)
    
#     state_data[i] = np.reshape(x_current, (13,))
#     control_data[i] = np.reshape(u_current, (4,))


    


# import sys
# sys.path.append('..')
# from plots import plot_state, plot_control
# # first we print out the state in one plot
# plot_state(tspan, state_data)
# plot_control(tspan, control_data)
# from animation import RocketAnimation
# rc = RocketAnimation([-1, -0.1, -0.2], [0,1,0], 0.4)
# rc.animate(tspan, state_data, control_data)
    