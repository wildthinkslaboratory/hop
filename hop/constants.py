import numpy as np

# TO DO:
# - Each constant could use a documentation line
# - Add missing data members to __repr__() and dictionary() functions

class Constants:
    def __init__(self):
        self.gx = 0
        self.gy = 0
        self.gz = -9.81
        self.m = 1
        self.l = 2           # moment arm length
        self.Ixx = 1
        self.Iyy = 1
        self.Izz = 1
        self.dt = 0.1

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
        

    
