import numpy as np
import casadi as ca
from hop.utilities import q_component_to_angle


class Constants:
    def __init__(self):

        # general constants
        # ---------------------------------------------------------------
        self.timelimit = 20.0 # time limit for a flight in seconds 
        self.shutdown_angle = 15.0 # shutdown if attitude exceeds this angle
        self.run_nmpc = True
        self.nmpc_delay = 3 # how many cycles it takes for the control to be actuated 

        self.battery_v = 25.0 # 25 volt battery
        self.v_alpha = 0.75 # factor for low pass filter of voltage

        # model related constants
        # ---------------------------------------------------------------
        self.m = 1.617    # mass of drone in kg

        self.px4_height = 0.3

        self.gx = 0     # acceleration due to gravity in world frame
        self.gy = 0
        self.gz = -9.81
        self.g = np.array([
            self.gx,
            self.gy,
            self.gz
        ])

        # self.Ixx =  0.0595     # moments of inertia
        # self.Iyy =  0.0598
        # self.Izz =  0.0128
        # self.Ixz =  0.0003
        # self.Iyz =  0.0010

        self.Ixx =  0.0621     # moments of inertia
        self.Iyy =  0.0624
        self.Izz =  0.0130
        self.Ixz =  0.0002
        self.Iyz =  0.0007


        self.I = np.array([
            [self.Ixx, 0.0,      self.Ixz],
            [0.0,      self.Iyy, self.Iyz],
            [self.Ixz, self.Iyz, self.Izz]
        ])

        # self.moment_arm = np.array([
        #     0.0015,
        #     0.007,
        #     -0.209799
        # ])

        self.moment_arm = np.array([
            0.000035,
            -0.000072,
            -0.21531
        ])



        self.I_diag_temp = [self.Ixx, self.Iyy, self.Izz]
        self.I_inv = np.linalg.inv(self.I)

        # thrust model and mapping
        # thrust is modeled as a degree 2 polynomial with coefficients a, b, c
        # that is scaled by a thrust curve constant
        self.tcc = 9.81 # thrust curve constant 
        self.a = 1.647 * self.tcc
        self.b = 0.9797 * self.tcc
        self.c = 0.03 * self.tcc
        # rotation about z axis caused by differential thrust between motors is modeled linearly with d
        self.d = 6.0
        self.thrust_constant = 1.3

 
        # mechanical and hardware constants
        # ---------------------------------------------------------------    
        self.gimbal_offset = [4.0, 0.0]   
        self.outer_gimbal_range = [-20,20]          # outer gimbal range limit in degrees
        self.inner_gimbal_range = [-13.5,13.5]      # inner gimbal range limit in degrees
        self.theta_dot_constraint = 6.16            # gimbal rate of change limit in degrees per dt
        self.thrust_dot_limit = 20.0                # thrust rate of change limit in Newtons per dt
        self.hover_thrust = 0.60                   # the thrust rate needed to hover
        self.prop_thrust_constraint = 1.0          # max thrust allowed 
        self.diff_thrust_constraint = [-0.2,0.2]    # min and max thrust difference allowed

        # NMPC related constants
        # ---------------------------------------------------------------        
        self.dt = 0.02 # 50 Hz like in paper
        self.x0 = ca.vertcat(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,1.0, 0.0,0.0,0.0) # initial state                                                    # state cost matrix

        # self.Q = ca.diag([80.0,80.0,100.0, 20.0,20.0,25.0, 2500.0,2500.0,200.0,200.0, 20.0,20.0,1.0 ])
        # self.Q = ca.diag([40.0,40.0,50.0, 10.0,10.0,15.0, 2500.0,2500.0,200.0,200.0, 30.0,30.0,1.0 ])

        # self.Q = ca.diag([50.0,50.0,50.0, 10.0,10.0,10.0, 526.0,526.0,15.0,0.0, 15.0,15.0,1.0 ])
        self.Q = ca.diag([10.0,10.0,10.0, 5.0,5.0,5.0, 526.0,526.0,33.0,0.0, 18.0,18.0,8.0 ])
        self.R = ca.diag([0.01, 0.01, 100, 100])
        
        self.gmb_deg_1pwm = 52

        # The JX PDI-6221MG servo has a speed of 0.18 sec/60° at 4.8V 
        # that's 6.5 degrees per 0.02 sec so moving 6 degrees in a time step would be max
        # gimbal angle degrees change per dt
        self.gmb_deg_dt = 6.0

        self.nmpc_rate_constraints = True

        # Ballpark guess, thrust is allowed to go from 0 to 1 in 0.5-1 second
        # that would mean a change of 0.02-0.04 per time step.
        # P average thrust change allowed per dt
        self.P_avg_dt = 0.04
        self.P_diff_dt = 0.02

        self.rate_scale_factor = 10
        self.actuator_rate_costs = self.rate_scale_factor * np.array([
            1, 
            1, 
            1.0/self.P_avg_dt, 
            1.0/self.P_diff_dt
        ])

        # control cost matrix
        self.terminal_cost_factor = 15.0
        self.xr = ca.vertcat(0.0,0.0,self.px4_height, 0.0,0.0,0.0, 0.0,0.0,0.0,1.0, 0.0,0.0,0.0) # goal state
        self.ur = ca.DM([0.0, 0.0, self.hover_thrust, 0.0])                          # goal control

        # list of navigation waypoints for the flight to follow
        # these are (x,y,z) points in world frame meters
        # self.waypoints = [
        #     np.array([0.0, 0.0, 0.3, 25.0, 0.0]),
        #     np.array([0.0, 0.0, 0.4, 25.0, self.hover_thrust]),    
        #     np.array([0.0, 0.0, 0.5, 25.0, self.hover_thrust]),
        #     np.array([0.0, 0.0, 0.6, 25.0, self.hover_thrust]),
        #     np.array([0.0, 0.1, 0.6, 25.0, self.hover_thrust]),
        #     np.array([0.0, 0.2, 0.6, 25.0, self.hover_thrust]),    
        #     np.array([0.0, 0.3, 0.6, 25.0, self.hover_thrust]),
        #     np.array([0.0, 0.0, 0.5, 25.0, self.hover_thrust]),
        #     np.array([0.0, 0.0, 0.4, 25.0, self.hover_thrust]),
        #     np.array([0.0, 0.0, 0.3, 25.0, self.hover_thrust])
        # ]

        self.waypoints = [
   
            # np.array([0.0, 0.0, 0.5, 25.0, self.hover_thrust]),
            # np.array([0.0, 0.0, 0.6, 25.0, self.hover_thrust]),
            # np.array([0.0, 0.0, 0.5, 25.0, self.hover_thrust]),
            np.array([0.0, 0.0, 0.9, 25.0, self.hover_thrust]),
            np.array([0.0, 0.0, 0.3, 25.0, 0.0]),
            np.array([0.0, 0.0, 0.3, 25.0, 0.0]),
        ]

        self.land = np.array([0.0, 0.0, self.px4_height, 23.0])



        # constants for specific NLP formulations
        # --------------------------------------------------------------- 

        self.horizon_time = 1.5

        # multiple shooter constants
        self.ms_time_step = 0.25 # number of timesteps for nmpc to consider

        # chebyshev pseudospectral constants
        self.spectral_order = 6

        # do-mpc constants
        self.finite_interval_size = 0.25
        self.number_intervals = 4
        self.collocation_degree = 2

        # IPOPT settings
        # --------------------------------------------------------------- 
        self.ipopt_settings = {
            "ipopt.max_iter": 50,                   
            "ipopt.tol": 1e-3,                     
            "ipopt.acceptable_tol": 1e-4,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.linear_solver': 'ma27',
            "ipopt.max_wall_time": 0.03,
            # 'ipopt.warm_start_init_point': 'yes',
            # 'ipopt.warm_start_bound_push': 1e-6,
            # 'ipopt.warm_start_mult_bound_push': 1e-6,
            # 'ipopt.mu_init': 1e-3,  
        }


    def tuning_info(self):
        s = 'Q Tuning Information\n'
        s += '-----------------------\n'
        s += 'position deviation\n'
        s += 'x: ' +  str(np.sqrt(1 / self.Q[0, 0])) + ' m\n'
        s += 'y: ' +  str(np.sqrt(1 / self.Q[1, 1])) + ' m\n'
        s += 'z: ' +  str(np.sqrt(1 / self.Q[2, 2])) + ' m\n'
        s += 'velocity deviation\n'
        s += 'vx: ' +  str(np.sqrt(1 / self.Q[3, 3])) + ' m / s\n'
        s += 'vy: ' +  str(np.sqrt(1 / self.Q[4, 4])) + ' m / s\n'
        s += 'vz: ' +  str(np.sqrt(1 / self.Q[5, 5])) + ' m / s\n'
        s += 'angle deviation\n'
        s += 'qx: ' +  str(round(q_component_to_angle(np.sqrt(1 / self.Q[6, 6])))) + ' degrees\n'
        s += 'qy: ' +  str(round(q_component_to_angle(np.sqrt(1 / self.Q[7, 7])))) + ' degrees\n'
        s += 'qz: ' +  str(round(q_component_to_angle(np.sqrt(1 / self.Q[8, 8])))) + ' degrees\n'
        s += 'angular velocity deviation\n'
        s += 'vx: ' +  str(np.sqrt(1 / self.Q[10, 10]) * (180.0 / np.pi)) + ' deg / s\n'
        s += 'vy: ' +  str(np.sqrt(1 / self.Q[11, 11]) * (180.0 / np.pi)) + ' deg / s\n'
        s += 'vz: ' +  str(np.sqrt(1 / self.Q[12, 12]) * (180.0 / np.pi)) + ' deg / s\n'
        s += '\nR Tuning Information\n'
        s += '-----------------------\n'
        s += 'gimbal deviation\n'
        s += 'theta 1: ' +  str(np.sqrt(1 / self.R[0, 0])) + ' degrees / sec\n'
        s += 'theta 2: ' +  str(np.sqrt(1 / self.R[1, 1])) + ' degrees / sec\n'
        s += 'P avg: ' +  str(np.sqrt(1 / self.R[2, 2])) + ' [0-1]\n'
        s += 'P diff: ' +  str(np.sqrt(1 / self.R[3, 3])) + ' [0-1]\n'
        return s



    def __dict__(self):
        mcd = {}
        mcd['battery_v'] = self.battery_v
        mcd['timelimit'] = self.timelimit
        mcd['shutdown_angle'] = self.shutdown_angle
        mcd['nmpc_delay'] = self.nmpc_delay
        mcd['m'] = self.m
        mcd['a'] = self.a
        mcd['b'] = self.b
        mcd['c'] = self.c
        mcd['d'] = self.d
        mcd['thrust_constant'] = self.thrust_constant
        mcd['px4_height'] = self.px4_height
        mcd['dt'] = self.dt
        mcd['terminal_cost_factor'] = self.terminal_cost_factor
        mcd['hover_thrust'] = self.hover_thrust 
        mcd['gmb_deg_dt'] = self.gmb_deg_dt 
        mcd['P_avg_dt'] = self.P_avg_dt 
        mcd['P_diff_dt'] = self.P_diff_dt           
        mcd['Q'] = ca.diag(self.Q).full().flatten().tolist()
        mcd['R'] = ca.diag(self.R).full().flatten().tolist()
        mcd['g'] = self.g.tolist()
        mcd['x0'] = self.x0.full().flatten().tolist()
        mcd['xr'] = self.xr.full().flatten().tolist()
        mcd['ur'] = self.ur.full().flatten().tolist()
        mcd['moment_arm'] = self.moment_arm.tolist()
        mcd['I'] = self.I.tolist()
        mcd['gimbal_offset'] = self.gimbal_offset 
        mcd['outer_gimbal_range'] = self.outer_gimbal_range 
        mcd['inner_gimbal_range'] = self.inner_gimbal_range 
        mcd['theta_dot_constraint'] = self.theta_dot_constraint
        mcd['thrust_dot_limit'] = self.thrust_dot_limit 
        mcd['prop_thrust_constraint'] = self.prop_thrust_constraint 
        mcd['diff_thrust_constraint'] = self.diff_thrust_constraint 
        mcd['gmb_deg_1pwm'] = self.gmb_deg_1pwm 
        mcd['nmpc_rate_constraints'] = self.nmpc_rate_constraints
        mcd['rate_scale_factor'] = self.rate_scale_factor
        mcd['actuator_rate_costs'] = self.actuator_rate_costs.tolist()
        mcd['horizon_time'] = self.horizon_time
        mcd['ms_time_step'] = self.ms_time_step  
        mcd['spectral_order'] = self.spectral_order
        mcd['finite_interval_size'] = self.finite_interval_size 
        mcd['number_intervals'] = self.number_intervals
        mcd['collocation_degree'] = self.collocation_degree
        mcd['ipopt_settings'] = self.ipopt_settings
        return mcd


    def update_from_dictionary(self, mcd):
        if 'battery_v' in mcd:
            self.battery_v = mcd['battery_v']
        if 'timelimit' in mcd:
            self.timelimit = mcd['timelimit']
        if 'shutdown_angle' in mcd:
            self.shutdown_angle = mcd['shutdown_angle']
        if 'nmpc_delay' in mcd:
            self.nmpc_delay = mcd['nmpc_delay']
        if 'm' in mcd:
            self.m = mcd['m']
        if 'a' in mcd:
            self.a = mcd['a'] 
        if 'b' in mcd:
            self.b = mcd['b'] 
        if 'c' in mcd:
            self.c = mcd['c'] 
        if 'd' in mcd:
            self.d = mcd['d'] 
        if 'thrust_constant' in mcd:
            self.thrust_constant = mcd['thrust_constant']
        if 'px4_height' in mcd:
            self.px4_height = mcd['px4_height']
        if 'dt' in mcd:
            self.dt = mcd['dt']
        if 'terminal_cost_factor' in mcd:
            self.terminal_cost_factor = mcd['terminal_cost_factor']
        if 'hover_thrust' in mcd:
            self.hover_thrust = mcd['hover_thrust']
        if 'gmb_deg_dt' in mcd:
            self.gmb_deg_dt = mcd['gmb_deg_dt']
        if 'P_avg_dt' in mcd:
            self.P_avg_dt = mcd['P_avg_dt']
        if 'P_diff_dt' in mcd:
            self.P_diff_dt = mcd['P_diff_dt']
        if 'Q' in mcd:
            self.Q = ca.diag(mcd['Q'])
        if 'R' in mcd:
            self.R = ca.diag(mcd['R'])
        if 'g' in mcd:
            self.g = np.array(mcd['g'])
        if 'x0' in mcd:
            self.x0 = ca.vertcat(mcd['x0'])
        if 'xr' in mcd:
            self.xr = ca.vertcat(mcd['xr'])
        if 'ur' in mcd:
            self.ur = ca.DM(mcd['ur'])
        if 'moment_arm' in mcd:
            self.moment_arm = np.array(mcd['moment_arm'])
        if 'I' in mcd:
            self.I = np.array(mcd['I'])
        if 'gimbal_offset' in mcd:
            self.gimbal_offset = mcd['gimbal_offset']
        if 'outer_gimbal_range' in mcd:
            self.outer_gimbal_range = mcd['outer_gimbal_range']
        if 'inner_gimbal_range' in mcd:
            self.inner_gimbal_range = mcd['inner_gimbal_range']
        if 'theta_dot_constraint' in mcd:
            self.theta_dot_constraint = mcd['theta_dot_constraint']
        if 'thrust_dot_limit' in mcd:
            self.thrust_dot_limit = mcd['thrust_dot_limit'] 
        if 'prop_thrust_constraint' in mcd:
            self.prop_thrust_constraint = mcd['prop_thrust_constraint']
        if 'diff_thrust_constraint' in mcd:
            self.diff_thrust_constraint = mcd['diff_thrust_constraint']
        if 'gmb_deg_1pwm' in mcd:
            self.gmb_deg_1pwm = mcd['gmb_deg_1pwm']
        if 'nmpc_rate_constraints' in mcd:
            self.nmpc_rate_constraints = mcd['nmpc_rate_constraints']
        if 'rate_scale_factor' in mcd:
            self.rate_scale_factor = mcd['rate_scale_factor']
        if 'actuator_rate_costs' in mcd:
            self.actuator_rate_costs = np.array(mcd['actuator_rate_costs'])
        if 'horizon_time' in mcd:
            self.horizon_time = mcd['horizon_time'] 
        if 'ms_time_step' in mcd:
            self.ms_time_step = mcd['ms_time_step']
        if 'spectral_order' in mcd:
            self.spectral_order = mcd['spectral_order']
        if 'finite_interval_size' in mcd:
            self.finite_interval_size = mcd['finite_interval_size']
        if 'number_intervals' in mcd:
            self.number_intervals = mcd['number_intervals']
        if 'collocation_degree' in mcd:
            self.collocation_degree = mcd['collocation_degree']
        if 'ipopt_settings' in mcd:
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
        s += f"{'gimbal offset:':20}  {str(self.gimbal_offset)}\n" 
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



        

    