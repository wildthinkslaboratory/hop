# Analysis tools for flight logs
#
from hop.drone_model import DroneModel
from hop.multiShooting import DroneNMPCMultiShoot
from hop.constants import Constants
from hop.utilities import import_data
from plots import plot_state
import numpy as np
import matplotlib.pyplot as plt
from hop.multiShooting import DroneNMPCMultiShoot
import sys


# read in logfile and time point to begin analyzing
log_file_name = './plotter_logs/current.json'
start_time = 0.0
if len(sys.argv) > 1:
    log_file_name = sys.argv[1]
    start_time = float(sys.argv[2])
    print(log_file_name, start_time)

# Import the flight data
log = import_data(log_file_name)   
flight_constants = log['constants'] 
data = log['run_data']

# read in the flight data
state_data = np.empty([len(data),13])
control_data = np.empty([len(data),4])
pwm_motors = np.empty([len(data),2])
pwm_servos = np.empty([len(data),2])
params = np.empty([len(data),5])
voltage = []
# collect all the data into arrays
for i, d in enumerate(data):
    state_data[i] = np.array(d['state'])
    control_data[i] = np.array(d['control'])
    voltage.append(d['voltage'])
    pwm_motors[i] = np.array(d['pwm_motors'])
    pwm_servos[i] = np.array(d['pwm_servos'])
    if len(d['parameters']) == 4:
        params[i] = np.array(d['parameters'] + [0.0])
    else:
        params[i] = np.array(d['parameters'])



# we often want to cut off the beginning of the data
# and start analyzing when the drone takes off
# here we set up all the data structures for storing 
# the analysis data
dt = 0.02
stop_index = int(start_time // dt)
len_used_data = len(data) - stop_index -1

# Truncate the data to start at the takeoff
state_data = state_data[stop_index+1:]
control_data = control_data[stop_index+1:]
voltage = voltage[stop_index+1:]
pwm_motors = pwm_motors[stop_index+1:]
pwm_servos = pwm_servos[stop_index+1:]
params = params[stop_index+1:-1]

tspan = np.arange(0, len_used_data * dt , dt)
full_params = np.zeros(9)
fitting_data = np.empty([len_used_data-1,39]) ## 39

# build up the fitting data
for i in range(len(state_data)-1):
    full_params[:5] = params[i]
    full_params[3] = voltage[i]
    fitting_data[i] = np.concatenate((state_data[i], state_data[i+1], control_data[i], full_params), axis=0)





import casadi as ca
from casadi import sin, cos

mc = Constants()
mc.update_from_dictionary(flight_constants)

# First create our state variables and control variables
p = ca.SX.sym('p', 3, 1)
v = ca.SX.sym('v', 3, 1)
q = ca.SX.sym('q', 4, 1)
w = ca.SX.sym('w', 3, 1)

parameters = ca.SX.sym('parameters', 9)
x = ca.vertcat(p,v,q,w)
u = ca.SX.sym('u', 4, 1)

# Now we build up the equations of motion and create a function
# for the system dynamics

I_mat = ca.DM(mc.I)

norm_P_avg = u[2] * parameters[3] / mc.battery_v
F = parameters[5] * norm_P_avg**2 + parameters[6] * norm_P_avg + parameters[7] 
M = parameters[8] * mc.Izz * u[3]

F_vector = F * ca.vertcat(
    sin((np.pi/180)*u[1]),
    -sin((np.pi/180)*u[0])*cos((np.pi/180)*u[1]),
    cos((np.pi/180)*u[0])*cos((np.pi/180)*u[1])
)

roll_moment = ca.vertcat(0, 0, M)
# M_vector = ca.cross(np.array([
#             parameters[11],
#             parameters[12],
#             parameters[13]
#         ]), F_vector) + roll_moment

M_vector = ca.cross(mc.moment_arm, F_vector) + roll_moment
angular_momentum = I_mat @ w

r_b2w = ca.vertcat(
    ca.horzcat(1 - 2*(x[7]**2 + x[8]**2), 2*(x[6]*x[7] - x[8]*x[9]), 2*(x[6]*x[8] + x[7]*x[9])),
    ca.horzcat(2*(x[6]*x[7] + x[8]*x[9]), 1 - 2*(x[6]**2 + x[8]**2), 2*(x[7]*x[8] - x[6]*x[9])),
    ca.horzcat(2*(x[6]*x[8] - x[7]*x[9]), 2*(x[7]*x[8] + x[6]*x[9]), 1 - 2*(x[6]**2 + x[7]**2)),
)

Q_omega = ca.vertcat(
    ca.horzcat(0, x[12], -x[11], x[10]),
    ca.horzcat(-x[12], 0, x[10], x[11]),
    ca.horzcat(x[11], -x[10], 0, x[12]),
    ca.horzcat(-x[10], -x[11], -x[12], 0)
)

q_full = x[6:10]
q_full = q_full / ca.norm_2(q_full)

RHS = ca.vertcat(
    v,
    (r_b2w @ F_vector) / mc.m + mc.g,
    0.5 * Q_omega @ q_full,
    ca.solve(I_mat, M_vector - ca.cross(w, angular_momentum))
)

# f is function that returns the change in state for a given state and control values
f = ca.Function('f', [x, u, parameters], [RHS])


# initial parameter guess
# [a,b,c,d]
theta_0 = np.array([1.647, 0.979, 0.03, 0.611])* 9.81

dt = mc.dt



def residuals(theta, fitting_data):
    # print('calling residuals')
    # print(theta, '\n')
    dt = 0.01

    residual_list = []
    for d in fitting_data:
        params = np.concatenate((d[30:35], theta), axis=0)
        state = d[:13]
        next_state = d[13:26]
        control = d[26:30]

        # predict the next state and compare with actual next state
        # predicted_state = state + dt * f(state, np.reshape(control, (4,1)), params)
        # error = next_state -  np.reshape(predicted_state, (13,))

        error = (next_state - state)/dt - np.reshape(f(state, np.reshape(control, (4,1)), params), (13,))
        # error[0:3]  /= 5.0     # position
        # error[3:6]  /= 5.0     # velocity
        # error[6:10] /= 1.0     # quaternion
        # error[10:13] /= 5.0    # angular velocity
        # error = np.concatenate([
        #     error[0:6],      # position + velocity
        #     error[10:13]     # angular velocity
        # ])        
        residual_list.extend(error[10:13])
    return residual_list

from scipy.optimize import least_squares


lower = [-20.0, -20.0, -20.0, -20.0]

upper = [20.0, 20.0, 20.0, 20.0]

result = least_squares(
    residuals,
    theta_0,
    args=(fitting_data,),
    method='trf',
    bounds=(lower, upper)  
)

# print("initial residual norm:",
#       np.linalg.norm(residuals(theta_0, fitting_data)))

# print("perturbed residual norm:",
#       np.linalg.norm(residuals(theta_0 + 1e-4, fitting_data)))

optimized_params = result.x
initial_norm = np.linalg.norm(residuals(theta_0, fitting_data))
final_norm = np.linalg.norm(residuals(result.x, fitting_data))

print("\nInitial parameters:", theta_0)
print("initial norm:", initial_norm)
print("\nOptimized parameters:", optimized_params)
print("final norm:", final_norm)





