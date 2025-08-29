import numpy as np
import casadi as ca

# TO DO:
# - Each constant could use a documentation line
# - Add missing data members to __repr__() and dictionary() functions

class Constants:
    def __init__(self):
        self.gx = 0
        self.gy = 0
        self.gz = -9.81
        self.m = 1.5
        self.l = 1        
        self.Ixx = 0.06
        self.Iyy = 0.06
        self.Izz = 0.012

        self.dt = 0.02 # 50 Hz like in paper
        self.mpc_horizon = 100 # number of timesteps for nmpc to consider

        self.spectral_order = 6

        self.timelimit = 1 # in seconds

        self.a = 0
        self.b = 2.5
        self.c = 0.1
        self.d = 0.1

        self.outer_gimbal_range = [-20,20]
        self.inner_gimbal_range = [-13.5,13.5]
        self.theta_dot_constraint = 6.16
        self.thrust_dot_limit = 35.0  # rate per second

        self.prop_thrust_constraint = 22.0
        self.diff_thrust_constraint = [-0.8,0.8]

        self.g = np.array([
            self.gx,
            self.gy,
            self.gz
        ])

        self.moment_arm = np.array([
            0,
            0,
            -self.l/2
        ])

        self.I = np.array([
            [self.Ixx,0,0],
            [0,self.Iyy,0],
            [0,0,self.Izz]
        ])

        self.I_diag = [self.Ixx, self.Iyy, self.Izz]
        self.I_inv = np.linalg.inv(self.I)

        self.x0 = ca.vertcat(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,1.0, 0.0,0.0,0.0)
        self.Q = ca.DM.eye(13)
        self.R = ca.DM.eye(4) * 0.03
        self.xr = ca.vertcat(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,1.0, 0.0,0.0,0.0)



    # This function makes it possible to print the Constants with print function
    # This way we can add our constants to our runs and simulation logs.
    def __repr__(self):
        s = 'Constants \n' + '---------------------\n'
        s += f"{'gx:':10}  {str(self.gx):15}\n"
        s += f"{'gy:':10}  {str(self.gy):15}\n"
        s += f"{'gz:':10}  {str(self.gz):15}\n"
        s += f"{'m:':10}  {str(self.m):15}\n"
        s += f"{'l:':10}  {str(self.l):15}\n"
        s += f"{'Ixx:':10}  {str(self.Ixx):15}\n"
        s += f"{'Iyy:':10}  {str(self.Iyy):15}\n"
        s += f"{'Izz:':10}  {str(self.Izz):15}\n"
        s += f"{'g:':10}  {str(self.g.tolist()):15}\n"
        return s

    # This function turns Constants into a dictionary so it can be easily dumped into a file
    # or read in from a file
    def __dict__(self):
        return {  
            'gx': self.gx,  
            'gy': self.gy,
            'gz': self.gz,
            'm' : self.m,
            'l' : self.l,       
            'Ixx': self.Ixx,
            'Iyy': self.Iyy,
            'Izz': self.Izz,
            'dt' : self.dt,
            'timelimit': self.timelimit,
            'a' : self.a,
            'b' : self.b,
            'c' : self.c,
            'd' : self.d,
            'outer_gimbal_range': self.outer_gimbal_range,
            'inner_gimbal_range': self.inner_gimbal_range,
            'theta_dot_constraint': self.theta_dot_constraint,
            'prop_thrust_constraint': self.prop_thrust_constraint,
            'diff_thrust_constraint': self.diff_thrust_constraint,
            'x0': self.x0.full().tolist(),
            'Q' : ca.diag(self.Q).full().tolist(),
            'xr': self.xr.full().tolist()
        }

        

    
