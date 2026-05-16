import casadi as ca
from casadi import sin, cos
import numpy as np


class DroneNMPCMultiShoot:
    def __init__(self, equations):
        self.mc = equations.mc
        self.N = int(self.mc.horizon_time / self.mc.ms_time_step)
        self.dt = self.mc.ms_time_step
        self.E = equations
        self.record_nlp_stats = False
    

    # In this function we build up the NMPC problem instance
    def build_nmpc_instance(self):

        X0 = ca.SX.sym('X0', self.size_x())   # these are variables representing our initial state
        U0 = ca.SX.sym('U0', self.size_u())
        
        P0 = ca.vertcat(X0, U0, self.E.p)

        # we make a copy of the state variables for each N+1 time steps
        X = ca.SX.sym('X', self.size_x(), self.N+1)    

        # we make a copy of the control variables for each N time steps
        U = ca.SX.sym('U', self.size_u(), self.N)       

        # We make one long list of all the optimization variables
        # all the state variables preceed all the control variables.
        self.opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        num_vars = self.opt_vars.numel()

        # now we add upper and lower bounds on the optimization variables
        # start with just negative infinity to positive infinity for everything
        self.lbx = -np.inf * np.ones(num_vars)
        self.ubx =  np.inf * np.ones(num_vars)

        n_x_vars = self.size_x() * (self.N+1)

        self.lbx[2: n_x_vars: self.size_x()] = 0     # keep z position above 0
        self.lbx[n_x_vars:   num_vars: self.size_u()] = self.mc.outer_gimbal_range[0]     # outer gimbal lower bound
        self.lbx[n_x_vars+1: num_vars: self.size_u()] = self.mc.inner_gimbal_range[0]     # inner gimbal lower bound
        self.ubx[n_x_vars:   num_vars: self.size_u()] = self.mc.outer_gimbal_range[1]     # outer gimbal upper bound
        self.ubx[n_x_vars+1: num_vars: self.size_u()] = self.mc.inner_gimbal_range[1]     # inner gimbal upper bound

        # g constraints contain an expression that is constrained by an upper and lower bound
        self.lbg = []   # will hold lower bounds for g constraints
        self.ubg = []   # will hold upper bounds for g constraints

        g = X[:, 0] - X0  
        self.lbg += [0.0]*int(g.numel())
        self.ubg += [0.0]*int(g.numel())

        self.cost = 0.0

        x_r = ca.vertcat(self.E.p[:3], self.mc.xr[3:])
        u_r = ca.vertcat(0.0, 0.0, self.E.p[4] * self.mc.battery_v / self.E.p[3], 0.0)

        for k in range(self.N):
            x_k = X[:, k]    # state at time step k
            u_k = U[:, k]  # control at time step k

            # here we build up the cost function by summing up the squared
            # error from the goal state over each time step
            state_error_cost = (x_k - x_r).T @ self.mc.Q @ (x_k - x_r)
            control_cost = (u_k - u_r).T @ self.mc.R @ (u_k - u_r)
            self.cost = self.cost + state_error_cost + control_cost

            # here we create the constraints that require the solution
            # to obey our system dynamics. We use Runge Kutta integration
            # and for each time step, we create a constraint that requires
            # the state at time k+1 to equal the system dynamics applied to the 
            # the state at time k.
            next_state = X[:, k+1]
            k1 = self.E.f(x_k, u_k, self.E.p)
            k2 = self.E.f(x_k + self.dt/2*k1, u_k, self.E.p)
            k3 = self.E.f(x_k + self.dt/2*k2, u_k, self.E.p)
            k4 = self.E.f(x_k + self.dt * k3, u_k, self.E.p)
            next_state_RK4 = x_k + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, next_state - next_state_RK4)
            self.lbg += [0.0]*int(next_state.numel())
            self.ubg += [0.0]*int(next_state.numel())

            # build up the upper thrust limit constraints         
            g   = ca.vertcat(g, u_k[2] + 0.5*u_k[3] - self.mc.prop_thrust_constraint)
            g   = ca.vertcat(g, u_k[2] - 0.5*u_k[3] - self.mc.prop_thrust_constraint)
            self.lbg += [-ca.inf]*2
            self.ubg += [0.0]*2

            if self.mc.nmpc_rate_constraints:
                if k < self.N-1:
                    next_u = U[:, k+1]  
                    g   = ca.vertcat(g, u_k[0] - next_u[0] - self.mc.theta_dot_constraint)
                    g   = ca.vertcat(g, u_k[1] - next_u[1] - self.mc.theta_dot_constraint)
                    self.lbg += [-ca.inf]*2
                    self.ubg += [0.0]*2

                    g   = ca.vertcat(g, u_k[2] - next_u[2] - self.mc.thrust_dot_limit)
                    self.lbg += [-ca.inf]
                    self.ubg += [0.0]


        x_N = X[:, self.N]             # final state
        e_N = x_N - x_r        # final error
        Qf  = self.mc.Q                     # terminal weight matrix (scale Q heavier)
        self.cost = self.cost + e_N.T @ Qf @ e_N


        # Now we set up the solver and do all of the options and parameters

        # dictionary for defining our solver
        nlp_prob = {
            'f': self.cost,
            'x': self.opt_vars,
            'g': g,
            'p': P0
        }

        # dictionary for our solver options
        opts = self.mc.ipopt_settings

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        # We create our initial guess for a solution.
        # Later we'll use the previous solution as our new solution guess.
        # repeat x_initial_guess across the horizon
        x_initial_guess = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        X_init = np.tile(np.array(x_initial_guess).reshape(-1,1), (1, self.N+1))
        
        U_init = np.tile(self.mc.ur, self.N)
        
        # glue this all together to make our initial guess
        self.init_guess = np.concatenate([X_init.reshape(-1, order='F'),
                                    U_init.reshape(-1, order='F')])
        
        # Here we initialize our stored solution to zeros
        self.sol_x = np.zeros(self.size_x() * (self.N+1))
        self.sol_u = np.zeros(self.size_u() * self.N)
        self.first_iteration = True


    # take the current state, control and parameters and 
    # compute the optimal control values
    def make_step(self, x, u, params):

        x = ca.vertcat(x,u,params)

        # if it's not the first iteration, use a warm start from previous solution.
        # we shift the trajectory forward by on time step and then just repeat
        # the last timestep twice
        if self.first_iteration:
            self.first_iteration = False
        else:
            if self.dt == 0.02:
                x_traj = np.concatenate([self.sol_x[self.size_x():], self.sol_x[self.size_x() * self.N:]])
                u_traj = np.concatenate([self.sol_u[self.size_u():], self.sol_u[self.size_u() * (self.N - 1):]])
                self.init_guess = np.concatenate([x_traj, u_traj])
            else:
                self.init_guess = np.concatenate([self.sol_x, self.sol_u])

        # Call the NMPC solver 
        sol = self.solver(x0=self.init_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=x)
        sol_opt = sol['x'].full().flatten()

        # save the solution for warm starts
        self.sol_x = sol_opt[:self.size_x() *(self.N+1)]
        self.sol_u = sol_opt[self.size_x() *(self.N+1):]

        # keep track of some accuracy measures from solving the nlp
        if self.record_nlp_stats:
            f_fun = ca.Function("f_fun", [self.opt_vars, self.E.p], [self.cost])
            cost = float(f_fun(sol_opt, params))
            self.solver_stats = {
                'status': self.solver.stats()['return_status'], 
                'cost': cost, 
            }

        return self.sol_u[:self.size_u()] # return the first control step


    def set_start_state(self, x0):
        self.x0 = x0

    def size_u(self):
        return self.E.u.size1()
    
    def size_x(self):
        return self.E.x.size1()



