# This is the code we used to find the time delay.
#

from hop.drone_model import DroneModel
from hop.play_model import DronePlayModel
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
from hop.utilities import quaternion_to_angle, import_data
import casadi as ca
import numpy as np
import statistics as stats
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib import colors
from plots import plot_state, plot_control, plot_pwm, plot_attitude, plot_parameters, plot_weighted_error_state, plot_weighted_error_control
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
parameters = np.empty([len(data),5])
voltage = []
# collect all the data into arrays
for i, d in enumerate(data):
    state_data[i] = np.array(d['state'])
    control_data[i] = np.array(d['control'])
    voltage.append(d['voltage'])
    pwm_motors[i] = np.array(d['pwm_motors'])
    pwm_servos[i] = np.array(d['pwm_servos'])
    if len(d['parameters']) == 4:
        parameters[i] = np.array(d['parameters'] + [0.0])
    else:
        parameters[i] = np.array(d['parameters'])


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
parameters = parameters[stop_index+1:-1]

# first we make a model
mc = Constants()

# update the constants with those used in the flight
mc.update_from_dictionary(flight_constants)

model = DroneModel(mc)  

# create an nmpc to compute the control
# We run the nmpc on the flight state and see if the computed control
# matches the flight control for each timestep
mpc = DroneNMPCdompc(mc.dt, model.model)
mpc.setup_cost()


# mc.moment_arm = np.array([
#     0.0015,
#     0.007,
#     -0.209799
# ])

# mc.Ixx =  0.0595     # moments of inertia
# mc.Iyy =  0.0598
# mc.Izz =  0.0128
# mc.Ixz =  0.0003
# mc.Iyz =  0.0010

# mc.I = np.array([
#     [mc.Ixx, 0.0,      mc.Ixz],
#     [0.0,      mc.Iyy, mc.Iyz],
#     [mc.Ixz, mc.Iyz, mc.Izz]
# ])

ms_mpc = DronePlayModel(mc)


# data structures for the things we want to plot
tspan = np.arange(0, len_used_data * dt , dt)
flight_model_error = np.empty([len_used_data-1,13])
dx_model_error = np.empty([len_used_data-1,13])
wx_data = np.empty([len_used_data-1,6])
state_change = np.empty([len_used_data-1,13])
model_change_state = np.empty([len_used_data-1,13])

control_data_computed = np.empty([len_used_data-1,4])
control_computed_diff = np.empty([len_used_data-1,4])
predicted_state = np.empty([len_used_data-1,13])
residual_state = np.empty([len_used_data,13])
residual_control = np.empty([len_used_data,4])
attitude = np.empty([len_used_data,3])
time_data = []
cost_data = []


# Now we set the initial state
x_init = ca.DM(state_data[0])
xr = mc.xr
mpc.set_start_state(x_init)
x0 = x_init

# these should be read in from logged constants
xrnp = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
urnp = np.array([0.0, 0.0, mc.hover_thrust, 0.0])
delay_steps = 0
thrust_delay = 0
servo_delay = 0

# run the simulation
for i in range(delay_steps,len(state_data)-2):

    # update parameters with current voltage
    parameters[i][3] = voltage[i]

    # # first run the nmpc on the state
    # mpc.set_waypoint(np.array(parameters[i]))
    # u0 = mpc.mpc.make_step(state_data[i])

    

    # # we want to check the nmpc results for the flight
    # control_data_computed[i] = np.reshape(u0, (4,))
    # control_computed_diff[i] = control_data[i] - control_data_computed[i]
    # cost_data.append(mpc.mpc.data['_aux'][-1][2])
    # if not mpc.mpc.solver_stats['return_status'] == 'Solve_Succeeded':
    #     print(mpc.mpc.solver_stats['return_status'])


    # turn quaternions into attitude
    # it's easier to read
    # q = state_data[6:10].copy()
    # q = np.reshape(state_data[i][6:10].copy(), (4,))
    # attitude[i] = quaternion_to_angle(q)
    

    # # compute the weighted squared error
    # state_error = state_data[i] - xrnp
    # control_error = control_data[i] - urnp
    # for j in range(len(state_error)):
    #     residual_state[i][j] = state_error[j] * mc.Q[j,j] * state_error[j]
    # residual_control[i] = np.absolute(control_error)
    # for j in range(len(control_error)):
    #     residual_control[i][j] = control_error[j] * mc.R[j,j] * control_error[j]

   # predict the next state and compare with actual next state
    u_delayed = np.array([control_data[i-servo_delay][0], 
                          control_data[i-servo_delay][1],
                          control_data[i-thrust_delay][2],
                          control_data[i-thrust_delay][3]
                          ])
    dx = ms_mpc.f(state_data[i],u_delayed, parameters[i])
       
    x0 = state_data[i] + mc.dt* dx
    dx = np.reshape(dx, (13,))

    flight_model_error[i] = state_data[i+1] -  np.reshape(x0, (13,))
    dx_model_error[i] = (state_data[i+1] - state_data[i]) -  np.reshape(dx, (13,))

    state_change[i] = state_data[i+1] - state_data[i]
    model_change_state[i] = dx * mc.dt

    predicted_state[i] = np.reshape(x0, (13,))



# # compute statistics for the timing of the nmpc calls
# mean_time = round(stats.mean(time_data),3)
# max_time = round(max(time_data),3) 
# print('mean time: ', mean_time)
# print('max time: ', max_time)

def plot_temp(tspan, data1, data2, title='unknown data'):
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['xtick.labelsize'] = 8
    fig, axs = plt.subplots(1)
    fig.set_figheight(8)
    fig.suptitle(title)


    axs.plot(tspan, data1[:,3])
    axs.plot(tspan, data2[:,3])



    plt.xlabel('Time')
    # plt.show()


def plot_w_dx(tspan, data1, data2, title='angular velocity dx*dt'):
    fig, axs = plt.subplots(3)
    fig.set_figheight(8)
    fig.suptitle(title)


    axs[0].plot(tspan, data1[:,10])
    axs[0].plot(tspan, data2[:,10])
    axs[0].set_ylabel('$w_x$')

    axs[1].plot(tspan, data1[:,11])
    axs[1].plot(tspan, data2[:,11])
    axs[1].set_ylabel('$w_y$')

    axs[2].plot(tspan, data1[:,12])
    axs[2].plot(tspan, data2[:,12])
    axs[2].set_ylabel('$w_z$')


    plt.xlabel('Time')
    plt.savefig("wxdt.pdf", format="pdf", bbox_inches="tight")

def plot_v_dx(tspan, data1, data2, title='velocity dx*dt'):
    fig, axs = plt.subplots(3)
    fig.set_figheight(8)
    fig.suptitle(title)


    axs[0].plot(tspan, data1[:,3])
    axs[0].plot(tspan, data2[:,3])
    axs[0].set_ylabel('$v_x$')

    axs[1].plot(tspan, data1[:,4])
    axs[1].plot(tspan, data2[:,4])
    axs[1].set_ylabel('$v_y$')

    axs[2].plot(tspan, data1[:,5])
    axs[2].plot(tspan, data2[:,5])
    axs[2].set_ylabel('$v_z$')


    plt.xlabel('Time')





plot_state(tspan[:-1], flight_model_error, 'flight state vs model predicted state error')
# plot_state(tspan[:-1], dx_model_error, 'change in state vs model dx error')
# plot_state(tspan[:-1], predicted_state, 'predicted state')
# plot_weighted_error_state(tspan, residual_state, 'state weighted squared errors')
# plot_weighted_error_control(tspan, residual_control, 'control weighted squared errors')
# plot_attitude(tspan, attitude, 'attitude')
plot_control(tspan, control_data, 'flight control data')
# plot_state(tspan, state_data, 'state')

plot_w_dx(tspan[:-1], model_change_state, state_change)
plot_v_dx(tspan[:-1], model_change_state, state_change)
plt.show()



