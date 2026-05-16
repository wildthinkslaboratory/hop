import casadi as ca
from casadi import sin, cos
import numpy as np

from hop.chebyshev import chebyshev_D, weights, cheb_nodes_weights, barycentric_resample_matrix, chebyshev_segments


class DroneNMPCwithCPS:
    def __init__(self, equations):
        self.mc = equations.mc
        self.T = self.mc.horizon_time
        self.N = self.mc.spectral_order
        self.E = equations
        self.record_nlp_stats = False

    # In this function we build up the NMPC problem instance
    # we can't build it until we know the goal state
    def build_nmpc_instance(self):

        X0 = ca.SX.sym('X0', self.size_x())            # initial state
        U0 = ca.SX.sym('U0', self.size_u())

        P0 = ca.vertcat(X0, U0, self.E.p)

        # we make a copy of the state variables for each N+1 time steps
        X = ca.SX.sym('X', self.size_x(), self.N+1)   
        # we make a copy of the control variables for each N time steps 
        U = ca.SX.sym('U', self.size_u(), self.N+1)   

        # We make one long list of all the optimization variables
        # all the state variables preceed all the control variables.
        self.opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        num_vars = self.opt_vars.numel()

        # now we add upper and lower bounds on the optimization variables
        # start with just negative infinity to positive infinity for everything
        self.lbx = -np.inf * np.ones(num_vars)
        self.ubx =  np.inf * np.ones(num_vars)

        n_x_vars = self.size_x() * (self.N+1)
        n_u_vars = self.size_u() * (self.N+1)

        self.lbx[2: n_x_vars: self.size_x()] = 0     # keep z position above 0
        self.lbx[n_x_vars:   num_vars: self.size_u()] = self.mc.outer_gimbal_range[0]     # outer gimbal lower bound
        self.lbx[n_x_vars+1: num_vars: self.size_u()] = self.mc.inner_gimbal_range[0]     # inner gimbal lower bound
        self.ubx[n_x_vars:   num_vars: self.size_u()] = self.mc.outer_gimbal_range[1]     # outer gimbal upper bound
        self.ubx[n_x_vars+1: num_vars: self.size_u()] = self.mc.inner_gimbal_range[1]     # inner gimbal upper bound
        self.lbx[n_x_vars+2: num_vars: self.size_u()] = 0     # thrust lower bound
        
        tau_2_time = (self.T/2)
        D = chebyshev_D(self.N)
        D_ca = ca.DM(D)

        w = weights(self.N)

        # g constraints contain an expression that is constrained by an upper and lower bound
        self.lbg = []   # will hold lower bounds for g constraints
        self.ubg = []   # will hold upper bounds for g constraints
    
        # equations of motion constraints
        g = X[:, 0] - X0
        self.lbg += [0.0]*int(g.numel())
        self.ubg += [0.0]*int(g.numel())

        # cost function
        self.cost = 0.0

        x_r = ca.vertcat(self.E.p[:3], self.mc.xr[3:])
        u_r = ca.vertcat(0.0, 0.0, self.E.p[4] * self.mc.battery_v / self.E.p[3], 0.0)
        
        for j in range(self.N + 1):
            x_k = X[:, j]
            u_k = U[:, j]

            # cost function
            state_cost = (x_k - x_r).T @ self.mc.Q @ (x_k - x_r)
            control_cost = (u_k - u_r).T @ self.mc.R @ (u_k - u_r)
            running_cost = state_cost + control_cost 
            self.cost = self.cost + w[j] * running_cost

            # dynamics constraints
            f_k = self.E.f(x_k, u_k, self.E.p)
            g = ca.vertcat(g, (D_ca[j,:] @ X.T).T - tau_2_time * f_k)
            self.lbg += [0.0]*int(self.size_x())
            self.ubg += [0.0]*int(self.size_x())


            # upper thrust limit constraints         
            g   = ca.vertcat(g, (u_k[2] + 0.5*u_k[3]) - self.mc.prop_thrust_constraint)
            g   = ca.vertcat(g, (u_k[2] - 0.5*u_k[3]) - self.mc.prop_thrust_constraint)
            self.lbg += [-ca.inf]*2
            self.ubg += [0.0]*2

            # -------------------------------------------------------

            # these are piece-wise rate constraints. There seem to be other ways to do rate constraints
            # in CPS, but I'm not currently sure I understand them and no how to encode them. More study needed.
            if self.mc.nmpc_rate_constraints:
                if j > 1 and j < (self.N - 1):
                        next_u = U[:, j+1]  
                        dt_j = self.T / self.N
                        g   = ca.vertcat(g, (u_k[0] - next_u[0]) / dt_j - self.mc.theta_dot_constraint)
                        g   = ca.vertcat(g, (u_k[1] - next_u[1]) / dt_j - self.mc.theta_dot_constraint)
                        self.lbg += [-ca.inf]*2
                        self.ubg += [0.0]*2

                        g   = ca.vertcat(g, (u_k[2] - next_u[2]) / dt_j - self.mc.thrust_dot_limit)
                        self.lbg += [-ca.inf]
                        self.ubg += [0.0]


        x_N = X[:, self.N]
        e_N = x_N - x_r
        Qf = self.mc.Q
        self.cost = self.cost + e_N.T @ Qf @ e_N

        self.cost = self.cost * tau_2_time

        # Now we set up the solver and do all of the options and parameters
        nlp_prob = {
            'f': self.cost,
            'x': self.opt_vars,
            'g': g,
            'p': P0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, self.mc.ipopt_settings)
        
        x_initial_guess = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        X_init = np.tile(np.array(x_initial_guess).reshape(-1,1), (1, self.N+1))
        U_init = np.tile(self.mc.ur, self.N+1)
        self.init_guess = np.concatenate([X_init.reshape(-1, order='F'), U_init.reshape(-1, order='F')])
        
        # Here we initialize our stored solution to zeros
        self.sol_x = np.zeros(n_x_vars)
        self.sol_u = np.zeros(n_u_vars)
        self.first_iteration = True


    def make_step(self, x, u, params):

        x = ca.vertcat(x,u,params)

        if self.first_iteration:
            self.first_iteration = False
        else:
            # construct our initial guess for warm starts
            # Shift amount for receding horizon:
            eps = 0.02     # time shift
            [tau,w] = cheb_nodes_weights(self.N,'second')
            S  = barycentric_resample_matrix(tau,w,eps)

            S_kron_x = np.kron(S, np.eye(self.size_x()))    # ((nx*m) × (nx*m))
            x_pred_flat = S_kron_x @ self.sol_x

            S_kron_u = np.kron(S, np.eye(self.size_u()))    # ((nx*m) × (nx*m))
            u_pred_flat = S_kron_u @ self.sol_u
            self.init_guess = np.concatenate([x_pred_flat, u_pred_flat])

        sol = self.solver(x0=self.init_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=x)
        sol_opt = sol['x'].full().flatten()
        self.sol_x = sol_opt[:self.size_x() * (self.N+1)]
        self.sol_u = sol_opt[self.size_x() * (self.N+1):]

        # keep track of some accuracy measures from solving the nlp
        if self.record_nlp_stats:
            f_fun = ca.Function("f_fun", [self.opt_vars, self.E.p], [self.cost])
            cost = float(f_fun(sol_opt, params))
            self.solver_stats = {
                'status': self.solver.stats()['return_status'], 
                'cost': cost, 
            }

        return self.sol_u[:self.size_u()]





    def set_start_state(self, x0):
        self.x0 = x0


    def size_u(self):
        return self.E.u.size1()
    
    def size_x(self):
        return self.E.x.size1()

