import casadi as ca
from casadi import sin, cos
import numpy as np

from hop.chebyshev import chebyshev_D, chebyshev_segments, weights, weights_paper

from hop.constants import Constants
mc = Constants()

class DroneNMPCwithCGL:
    def __init__(self):

        self.T = mc.mpc_horizon * mc.dt
        self.N = mc.spectral_order

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

        X0 = ca.SX.sym('X0', self.x.size1())            # initial state
        U0 = ca.SX.sym('U0', self.size_u())
        P0 = ca.vertcat(X0, U0)

        X = ca.SX.sym('X', self.x.size1(), self.N+1)    

        U = ca.SX.sym('U', self.size_u(), self.N+1)   

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        num_vars = opt_vars.numel()

        self.lbx = -np.inf * np.ones(num_vars)
        self.ubx =  np.inf * np.ones(num_vars)

        n_x_vars = self.size_x() * (self.N+1)
        n_u_vars = self.size_u() * (self.N+1)

        self.lbx[2: n_x_vars: self.size_x()] = 0     # keep z position above 0


        self.lbx[n_x_vars:   num_vars: self.size_u()] = mc.outer_gimbal_range[0]     # outer gimbal lower bound
        self.lbx[n_x_vars+1: num_vars: self.size_u()] = mc.inner_gimbal_range[0]     # inner gimbal lower bound

        # self.lbx[n_x_vars+2: num_vars: self.size_u()] = 0                            # average thrust lower bound

        self.lbx[n_x_vars+3: num_vars: self.size_u()] = mc.diff_thrust_constraint[0] # delta thrust lower bound

        self.ubx[n_x_vars:   num_vars: self.size_u()] = mc.outer_gimbal_range[1]     # outer gimbal upper bound
        self.ubx[n_x_vars+1: num_vars: self.size_u()] = mc.inner_gimbal_range[1]     # inner gimbal upper bound
        self.ubx[n_x_vars+3: num_vars: self.size_u()] = mc.diff_thrust_constraint[1] # delta thrust upper bound


        tau_2_time = (self.T/2)
        D = chebyshev_D(self.N)
        D_ca = ca.DM(D)

        w = weights_paper(self.N)


        # g constraints contain an expression that is constrained by an upper and lower bound
        self.lbg = []   # will hold lower bounds for g constraints
        self.ubg = []   # will hold upper bounds for g constraints
    

        # equations of motion constraints
        g = X[:, 0] - X0
        self.lbg += [0.0]*int(g.numel())
        self.ubg += [0.0]*int(g.numel())

        first_u = U[:, 0]  
        g   = ca.vertcat(g, (first_u[2] + 0.5*first_u[3]) - (U0[2] + 0.5*U0[3]))
        g   = ca.vertcat(g, (first_u[2] - 0.5*first_u[3]) - (U0[2] - 0.5*U0[3]))
        self.lbg += [-mc.thrust_dot_limit * mc.dt] * 2
        self.ubg += [mc.thrust_dot_limit * mc.dt] * 2

        # cost function
        cost = 0.0
        for j in range(self.N + 1):
            x_k = X[:, j]
            u_k = U[:, j]
            error = x_k - self.x_goal

            state_cost = (error).T @ mc.Q @ (error)
            control_cost = u_k.T @ mc.R @ u_k
            running_cost = state_cost + control_cost
            cost = cost + w[j] * running_cost


            # dynamics constraints
            f_k = self.f(x_k, u_k)
            g = ca.vertcat(g, (D_ca[j,:] @ X.T).T - tau_2_time * f_k)
            self.lbg += [0.0]*int(self.size_x())
            self.ubg += [0.0]*int(self.size_x())

            # build up the upper thrust limit constraints         
            g   = ca.vertcat(g, (u_k[2] + 0.5*u_k[3]) - mc.prop_thrust_constraint)
            g   = ca.vertcat(g, (u_k[2] - 0.5*u_k[3]) - mc.prop_thrust_constraint)
            self.lbg += [-ca.inf]*2
            self.ubg += [0.0]*2

            # # build up rate of change constraints for thrust 
            # if j < self.N-1:
            #     next_u = U[:, j+1]  
            #     g   = ca.vertcat(g, (next_u[2] + 0.5*next_u[3]) - (u_k[2] + 0.5*u_k[3]))
            #     g   = ca.vertcat(g, (next_u[2] - 0.5*next_u[3]) - (u_k[2] - 0.5*u_k[3]))
            #     self.lbg += [-mc.thrust_dot_limit * dt_segs[j]] * 2
            #     self.ubg += [mc.thrust_dot_limit * dt_segs[j]] * 2


        cost = cost * tau_2_time
        x_N = X[:, self.N]
        e_N = x_N - self.x_goal
        Qf = 10*mc.Q
        cost = cost + e_N.T @ Qf @ e_N

        R0 = ca.diag(ca.DM([1, 1, 2, 1]))
        cost_first_move = (U[:, 0] - U0).T @ R0 @ (U[:, 0] - U0)
        cost = cost + cost_first_move


        # Now we set up the solver and do all of the options and parameters
        nlp_prob = {
            'f': cost,
            'x': opt_vars,
            'g': g,
            'p': P0
        }

        opts = {
            'ipopt.max_iter': 200,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        x_initial_guess = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        X_init = np.tile(np.array(x_initial_guess).reshape(-1,1), (1, self.N+1))

        # initial guess for controls is zero
        U_init = np.zeros((self.size_u(), self.N+1))

        # U_initial_guess = ca.DM([0.0,0.0,5.67,0.0])
        # U_init = np.tile(np.array(U_initial_guess).reshape(-1,1), (1, self.N+1))

        # glue this all together to make our initial guess
        self.init_guess = np.concatenate([X_init.reshape(-1, order='F'), U_init.reshape(-1, order='F')])
        
        # Here we initialize our stored solution to zeros
        self.sol_x = np.zeros(n_x_vars)
        self.sol_u = np.zeros(n_u_vars)
        self.first_iteration = True


    def make_step(self, x, u):

        x = ca.vertcat(x,u)

        if self.first_iteration:
            self.first_iteration = False
        else:
            x_traj = np.concatenate([self.sol_x[self.size_x():], self.sol_x[self.size_x() * self.N:]])
            u_traj = np.concatenate([self.sol_u[self.size_u():], self.sol_u[self.size_u() * (self.N):]])
            self.init_guess = np.concatenate([x_traj, u_traj])
        sol = self.solver(x0=self.init_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=x)
        sol_opt = sol['x'].full().flatten()

        self.sol_x = sol_opt[:self.size_x() * (self.N+1)]
        self.sol_u = sol_opt[self.size_x() * (self.N+1):]

        # reshape U vector back to (nu, N+1) in Fortran order
        Umat = np.reshape(self.sol_u, (self.size_u(), self.N+1), order='F')
        # simple, effective approximation: average node 0 and 1
        self.sol_u[:self.size_u()] = 0.5*(Umat[:, 0] + Umat[:, 1])

        return self.sol_u[:self.size_u()]



    def set_start_state(self, x0):
        self.x0 = x0


    def size_u(self):
        return self.u.size1()
    
    def size_x(self):
        return self.x.size1()



# solver = DroneNMPCCasadi()

# sim_time = 5.0
# n_sim_steps = int(sim_time/mc.dt)
# tspan = np.arange(0, n_sim_steps * mc.dt, mc.dt)

# # The starting state
# x_current = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0]) 

# state_data = np.empty([n_sim_steps,13])
# control_data = np.empty([n_sim_steps,4])

# state_goal = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
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
    