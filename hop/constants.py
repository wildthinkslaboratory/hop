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
        self.m = 1
        self.l = 2          
        self.Ixx = 1
        self.Iyy = 1
        self.Izz = 1
        self.dt = 0.1
        self.timelimit = 1 # in seconds

        self.a = 0
        self.b = 1
        self.c = 0
        self.d = 1

        self.outer_gimbal_range = [-15,15]
        self.inner_gimbal_range = [-15,15]

        self.theta_dot_constraint = [-6.16,6.16]

        self.prop_thrust_constraint = [0,1.0]
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

        self.x0 = ca.vertcat(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0)
        self.Q = ca.diag(13)
        self.xr = ca.vertcat(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0)



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
            'g': self.g.tolist(),
        }

        

    
