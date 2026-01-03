import numpy as np
import casadi as ca

class Constants:
    def __init__(self):

        # general constants
        # ---------------------------------------------------------------
        self.timelimit = 1 # time limit for a flight in seconds 

        self.battery_v = 25.0 # 25 volt battery

        # model related constants
        # ---------------------------------------------------------------
        self.m = 1.584    # mass of drone in kg

        self.px4_height = 0.3

        self.gx = 0     # acceleration due to gravity in world frame
        self.gy = 0
        self.gz = -9.81
        self.g = np.array([
            self.gx,
            self.gy,
            self.gz
        ])

        self.Ixx =  0.0595     # moments of inertia
        self.Iyy =  0.0598
        self.Izz =  0.0128
        self.Ixz =  0.0003
        self.Iyz =  0.0010


        self.I = np.array([
            [self.Ixx, 0.0,      self.Ixz],
            [0.0,      self.Iyy, self.Iyz],
            [self.Ixz, self.Iyz, self.Izz]
        ])

        self.moment_arm = np.array([
            0.0015,
            0.007,
            -0.209799
        ])



        self.I_diag_temp = [self.Ixx, self.Iyy, self.Izz]
        self.I_inv = np.linalg.inv(self.I)

        # thrust model and mapping
        # thrust is modeled as a degree 2 polynomial with coefficients a, b, c
        # that is scaled by a thrust curve constant
        self.tcc = 9.81 # thrust curve constant 
        # these values are based on the 21V data from 
        # https://drive.google.com/file/d/1KMV0z-SipDZAr_uxRndRSnhmOHLBXNA5/view
        self.a = 1.647 * self.tcc
        self.b = 0.9797 * self.tcc
        self.c = 0.03 * self.tcc
        # rotation about z axis caused by differential thrust between motors is modeled linearly with d
        self.d = 6.0

 
        # mechanical and hardware constants
        # ---------------------------------------------------------------       
        self.outer_gimbal_range = [-20,20]          # outer gimbal range limit in degrees
        self.inner_gimbal_range = [-13.5,13.5]      # inner gimbal range limit in degrees
        self.theta_dot_constraint = 6.16            # gimbal rate of change limit in degrees per second
        self.thrust_dot_limit = 20.0                # thrust rate of change limit in Newtons per second
        self.hover_thrust = 0.72                   # the thrust rate needed to hover
        self.prop_thrust_constraint = 1.0          # max thrust allowed 
        self.diff_thrust_constraint = [-0.2,0.2]    # min and max thrust difference allowed

        # NMPC related constants
        # ---------------------------------------------------------------        
        self.dt = 0.02 # 50 Hz like in paper
        self.x0 = ca.vertcat(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,1.0, 0.0,0.0,0.0) # initial state                                                    # state cost matrix
    


        self.Q = ca.diag([40.0,40.0,50.0, 10.0,10.0,15.0, 2500.0,2500.0,200.0,200.0, 30.0,30.0,1.0 ])
        # previous divided by 40.0 to match with rate constaints
        # self.Q = ca.diag([1.0,1.0,1.25, 0.25,0.25,0.375, 62.0,62.0,5.0,5.0, 0.75,0.75,0.25 ])


        # these are the maximum allowed divergences from the goal state beyond which we want to discourage
        # with the cost function
        # self.pos_max = 1.0 # meters
        # self.vel_max = 1.0 # m / s
        # self.q_xy_max = 0.1 # quaternion [-1, 1] this is about 11 degrees
        # self.q_z_max = 0.7 # quaternion this is about 90 degrees
        # self.w_max = 100 # degrees / second

        # These things are all equally bad
        self.pos_max = 0.16  # meters
        self.vel_max = 0.31  # m / s
        self.q_xy_max = 0.02 # quaternion [-1, 1] this is about 2.3 degrees
        self.q_z_max = 0.07  # quaternion this is about 8 degrees
        self.w_max = 0.18    # degrees / second
        self.w_z_max = 1.0   # degrees / second

        self.Q = ca.diag([
            1/self.pos_max**2,
            1/self.pos_max**2,
            1/self.pos_max**2, 
            1/self.vel_max**2,
            1/self.vel_max**2,
            1/self.vel_max**2,
            1/self.q_xy_max**2,
            1/self.q_xy_max**2,
            1/self.q_z_max**2,
            1/self.q_z_max**2,
            1/self.w_max**2,
            1/self.w_max**2,
            1/self.w_z_max**2
        ])

        self.R = ca.diag([0.5, 0.5, 1, 0.03])

        # these are the maximum values for control beyond which we want to discourage
        self.gmb_max = 1.41   # in degrees
        self.P_avg_max = 1.0  # scale of 0 - 1
        self.P_diff_max = 5.77 # scale of 0 - 1

        self.R = ca.diag([
            1/self.gmb_max**2, 
            1/self.gmb_max**2, 
            1/self.P_avg_max**2, 
            1/self.P_diff_max**2
        ])


        # The JX PDI-6221MG servo has a speed of 0.18 sec/60° at 4.8V 
        # that's 6.5 degrees per 0.02 sec so moving 6 degrees in a time step would be max
        # gimbal angle degrees change per dt
        self.gmb_deg_dt = 6.0

        # Ballpark guess, thrust is allowed to go from 0 to 1 in 0.5-1 second
        # that would mean a change of 0.02-0.04 per time step.
        # P average thrust change allowed per dt
        self.P_avg_dt = 0.04
        self.P_diff_dt = 0.02

        self.actuator_rate_costs = np.array([
            1.0/self.gmb_deg_dt, 
            1.0/self.gmb_deg_dt, 
            1.0/self.P_avg_dt, 
            1.0/self.P_diff_dt
        ])

        # the terminal Q matrix places tighter control on angular momentum
        self.w_max_term = 1.0 # degrees / second
        self.w_z_max_term = 1.0 # degrees / second

        self.Q_term = self.Q
        self.Q_term[10:13] = np.array([
            1/self.w_max_term**2,
            1/self.w_max_term**2,
            1/self.w_max_term**2
        ])

        # control cost matrix
        self.terminal_cost_factor = 2.0
        self.xr = ca.vertcat(0.0,0.0,self.px4_height, 0.0,0.0,0.0, 0.0,0.0,0.0,1.0, 0.0,0.0,0.0) # goal state
        self.ur = ca.DM([0.0, 0.0, self.hover_thrust, 0.0])                          # goal control

        # list of navigation waypoints for the flight to follow
        # these are (x,y,z) points in world frame meters
        self.waypoints = [
            np.array([0.0, 0.0, 1.1, 25.0]),
            np.array([0.0, 0.0, 1.1, 25.0]),    
            np.array([0.0, 0.0, 0.5, 25.0]),
            np.array([0.0, 0.0, 0.5, 25.0])
        ]

        self.land = np.array([0.0, 0.0, self.px4_height, 23.0])

        self.nmpc_rate_constraints = False

        # constants for specific NLP formulations
        # --------------------------------------------------------------- 
        # multiple shooter constants
        self.mpc_horizon = 100 # number of timesteps for nmpc to consider

        # chebyshev pseudospectral constants
        self.spectral_order = 6

        # do-mpc constants
        self.finite_interval_size = 0.3
        self.number_intervals = 6
        self.collocation_degree = 2

        # IPOPT settings
        # --------------------------------------------------------------- 
        self.ipopt_settings = {
            "ipopt.max_iter": 100,                   
            "ipopt.tol": 1e-3,                     
            "ipopt.acceptable_tol": 1e-4,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.linear_solver': 'ma27',
            # 'ipopt.warm_start_init_point': 'yes',
            # 'ipopt.warm_start_bound_push': 1e-6,
            # 'ipopt.warm_start_mult_bound_push': 1e-6,
            # 'ipopt.mu_init': 1e-3,  
        }



    def __dict__(self):
        mcd = {}
        mcd['battery_v'] = self.battery_v
        mcd['m'] = self.m
        mcd['a'] = self.a
        mcd['b'] = self.b
        mcd['c'] = self.c
        mcd['d'] = self.d
        mcd['px4_height'] = self.px4_height
        mcd['dt'] = self.dt
        mcd['terminal_cost_factor'] = self.terminal_cost_factor
        mcd['hover_thrust'] = self.hover_thrust
        mcd['pos_max'] = self.pos_max = 0.16  # meters
        mcd['vel_max'] = self.vel_max = 0.31  # m / s
        mcd['q_xy_max'] = self.q_xy_max = 0.02 # quaternion [-1, 1] this is about 2.3 degrees
        mcd['q_z_max'] = self.q_z_max = 0.07  # quaternion this is about 8 degrees
        mcd['w_max'] = self.w_max = 0.18    # degrees / second
        mcd['w_z_max'] = self.w_z_max = 1.0   # degrees / second
        mcd['gmb_max'] = self.gmb_max = 1.41   # in degrees
        mcd['P_avg_max'] = self.P_avg_max = 1.0  # scale of 0 - 1
        mcd['P_diff_max'] = self.P_diff_max = 5.77 # scale of 0 - 1
        mcd['gmb_deg_dt'] = self.gmb_deg_dt = 6.0
        mcd['P_avg_dt'] = self.P_avg_dt = 0.04
        mcd['P_diff_dt'] = self.P_diff_dt = 0.02   
        mcd['w_max_term'] = self.w_max_term = 1.0 # degrees / second
        mcd['w_z_max_term'] = self.w_z_max_term = 1.0 # degrees / second             
        mcd['Q'] = ca.diag(self.Q).full().flatten().tolist()
        mcd['R'] = ca.diag(self.R).full().flatten().tolist()
        mcd['g'] = self.g.tolist()
        mcd['x0'] = self.x0.full().flatten().tolist()
        mcd['xr'] = self.xr.full().flatten().tolist()
        mcd['ur'] = self.ur.full().flatten().tolist()
        mcd['moment_arm'] = self.moment_arm.tolist()
        mcd['I'] = self.I.tolist()
        mcd['ipopt_settings'] = self.ipopt_settings
        return mcd


    def update_from_dictionary(self, mcd):
        self.battery_v = mcd['battery_v']
        self.m = mcd['m']
        self.a = mcd['a'] 
        self.b = mcd['b'] 
        self.c = mcd['c'] 
        self.d = mcd['d'] 
        self.px4_height = mcd['px4_height']
        self.dt = mcd['dt']
        self.hover_thrust = mcd['hover_thrust']
        self.Q = ca.diag(mcd['Q'])
        self.R = ca.diag(mcd['R'])
        self.g = np.array(mcd['g'])
        self.x0 = ca.vertcat(mcd['x0'])
        self.xr = ca.vertcat(mcd['xr'])
        self.ur = ca.DM(mcd['ur'])
        self.moment_arm = np.array(mcd['moment_arm'])
        self.I = np.array(mcd['I'])
        self.ipopt_settings = mcd['ipopt_settings']

    # This function makes it possible to print the Constants with print function
    # This way we can add our constants to our runs and simulation logs.
    def __repr__(self):
        s = 'Constants \n' + '---------------------\n'
        s += 'General constants: \n'
        s += '-----------------------------------------------\n'
        s += f"{'flight time:':15}  {str(self.timelimit):15}\n"
        s += 'Model related constants: \n'
        s += '-----------------------------------------------\n'
        s += f"{'m:':10}  {str(self.m):15}\n"
        s += f"{'gx:':10}  {str(self.gx):15}\n"
        s += f"{'gy:':10}  {str(self.gy):15}\n"
        s += f"{'gz:':10}  {str(self.gz):15}\n"
        s += f"{'g:':10}  {str(self.g.tolist()):15}\n"
        s += f"{'Ixx:':10}  {str(self.Ixx):15}\n"
        s += f"{'Iyy:':10}  {str(self.Iyy):15}\n"
        s += f"{'Izz:':10}  {str(self.Izz):15}\n"
        s += f"{'moment arm:':20}  {str(self.moment_arm.tolist())}\n" 
        s += f"{'I_inv:':20}  {str(self.I_inv.tolist())}\n" 
        s += 'thrust model constants: \n'
        s += '-----------------------------------------------\n'        
        s += f"{'tcc:':10}  {str(self.tcc):15}\n"
        s += f"{'a:':10}  {str(self.a):15}\n"
        s += f"{'b:':10}  {str(self.b):15}\n"
        s += f"{'c:':10}  {str(self.c):15}\n"
        s += f"{'d:':10}  {str(self.d):15}\n"
        s += 'Mechanical and hardware constants: \n'
        s += '-----------------------------------------------\n'
        s += f"{'outer gimbal range:':20}  {str(self.outer_gimbal_range)}\n" 
        s += f"{'inner gimbal range:':20}  {str(self.inner_gimbal_range)}\n" 
        s += f"{'theta dot max:':20}  {str(self.theta_dot_constraint)}\n" 
        s += f"{'thrust dot max:':20}  {str(self.thrust_dot_limit)}\n" 
        s += f"{'hover thrust:':20}  {str(self.hover_thrust)}\n" 
        s += f"{'max thrust:':20}  {str(self.prop_thrust_constraint)}\n" 
        s += f"{'max diff thrus:':20}  {str(self.diff_thrust_constraint)}\n" 
        s += 'NMPC constants: \n'
        s += '-----------------------------------------------\n'
        s += f"{'dt:':10}  {str(self.dt):15}\n"
        s += f"{'x0:':10}  {str(self.x0)}\n" 
        s += f"{'Q:':10}  {str(self.Q)}\n" 
        s += f"{'R:':10}  {str(self.R)}\n" 
        s += f"{'xr:':10}  {str(self.xr)}\n"
        s += f"{'ur:':10}  {str(self.ur)}\n"        
        s += f"{'waypoints:':20}  {str(self.waypoints):15}\n"   
        s += f"{'NMPC rate constraints:':20}  {str(self.nmpc_rate_constraints)}\n"  
        s += 'NLP constants: \n'
        s += '-----------------------------------------------\n'
        s += f"{'nmpc horizon:':20}  {str(self.mpc_horizon):15}\n"
        s += f"{'spectral order:':20}  {str(self.spectral_order):15}\n"
        s += f"{'size of intervals:':20}  {str(self.finite_interval_size):15}\n"
        s += f"{'num intervals:':20}  {str(self.number_intervals):15}\n"
        s += f"{'collocation deg:':20}  {str(self.collocation_degree):15}\n"
        s += 'IPOPT settings: \n'
        s += '-----------------------------------------------\n'
        s += str(self.ipopt_settings)

        return s



        

    