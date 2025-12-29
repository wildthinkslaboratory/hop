# Experiementing with which drone constants affect stability and performance
#
from hop.drone_model import DroneModel
from hop.multiShooting import DroneNMPCMultiShoot
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
from do_mpc.simulator import Simulator
from utilities import import_data
import casadi as ca
from do_mpc.estimator import StateFeedback
import numpy as np
import statistics as stats
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib import colors
from plots import plot_state, plot_control, plot_pwm
from hop.multiShooting import DroneNMPCMultiShoot

import sys

log_file_name = './plotter_logs/current.json'
start_time = 0.0
if len(sys.argv) > 1:
    log_file_name = sys.argv[1]
    start_time = float(sys.argv[2])
    print(log_file_name, start_time)

# first we make a model
mc = Constants()
model = DroneModel(mc)  

# now we need two nmpc algorithms
# the first is the one used during the live flight
# the second just has a handy function to compute the next
# state from the model.
mpc = DroneNMPCdompc(mc.dt, model.model)

mc.a = mc.a * 1.5
mc.b = mc.b * 1.5

ms_mpc = DroneNMPCMultiShoot(mc)

# we create dompc simulator to for the cumulative next state
estimator = StateFeedback(model.model)
sim = Simulator(model.model)
sim.set_param(t_step = mc.dt)

parameters = np.array([0.0, 0.0, 0.0, 22.0])
p_template = sim.get_p_template()
def dummy(t_now):
    p_template['parameters'] = parameters
    return p_template
sim.set_p_fun(dummy)

sim.setup()
mpc.setup_cost()


# Import the flight data
log = import_data(log_file_name)    
dt = 0.02
data = log['run_data']

# we often want to cut off the beginning of the data
# and start analyzing when the drone takes off
# here we set up all the data structures for storing 
# the analysis data
stop_index = int(start_time // dt)
len_used_data = len(data) - stop_index -1

tspan = np.arange(0, len_used_data * dt , dt)
control_data_computed = np.empty([len_used_data-1,4])
error = np.empty([len_used_data-1,13])
residual_state = np.empty([len_used_data,13])
residual_control = np.empty([len_used_data,4])
cum_error = np.empty([len_used_data-1,13])
voltage = []

state_data = np.empty([len(data),13])
control_data = np.empty([len(data),4])
pwm_motors = np.empty([len(data),2])
pwm_servos = np.empty([len(data),2])
# collect all the data into arrays
for i, d in enumerate(data):
    state_data[i] = np.array(d['state'])
    control_data[i] = np.array(d['control'])
    voltage.append(d['voltage'])
    pwm_motors[i] = np.array(d['pwm_motors'])
    pwm_servos[i] = np.array(d['pwm_servos'])

# Truncate the data to start at the takeoff
state_data = state_data[stop_index+1:]
control_data = control_data[stop_index+1:]
voltage = voltage[stop_index+1:]
pwm_motors = pwm_motors[stop_index+1:]
pwm_servos = pwm_servos[stop_index+1:]

# Now we set the initial state
x_init = ca.DM(state_data[0])
xr = mc.xr

mpc.set_start_state(x_init)
sim.x0 = x_init
estimator.x0 = x_init

x0 = x_init
time_data = []
cost_data = []
xrnp = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
urnp = np.array([0.0, 0.0, mc.hover_thrust, 0.0])
waypoint = [0.0, 0.0, 0.7, 25.0]
# run the simulation
for i in range(len(state_data)-1):

    # first run the nmpc on the state
    waypoint[3] = voltage[i]
    start_time = perf_counter()
    mpc.set_waypoint(np.array(waypoint))
    u0 = mpc.mpc.make_step(state_data[i])
    step_time = perf_counter() - start_time

    control_data_computed[i] = np.reshape(u0, (4,))
    
    state_error = state_data[i] - xrnp
    control_error = control_data[i] - urnp
    residual_state[i] = np.absolute(state_error)
    residual_control[i] = np.absolute(control_error)

    y_next = sim.make_step(np.reshape(control_data[i], (4,1)))
    x_cum = estimator.make_step(y_next)

    x0 = state_data[i] + mc.dt* ms_mpc.f(state_data[i],np.reshape(control_data[i], (4,1)))

    error[i] = np.reshape(x0, (13,)) - state_data[i+1]
    cum_error[i] = np.reshape(x_cum, (13,)) - state_data[i+1]

    time_data.append(step_time)
    cost_data.append(mpc.mpc.data['_aux'][-1][2])
    if not mpc.mpc.solver_stats['return_status'] == 'Solve_Succeeded':
        print(mpc.mpc.solver_stats['return_status'])



# compute statistics for the timing of the nmpc calls
mean_time = round(stats.mean(time_data),3)
max_time = round(max(time_data),3) 
print('mean time: ', mean_time)
print('max time: ', max_time)

# now all the plots
plt.figure(figsize=(6,3))

plt.figure(1)
plt.plot(tspan[:-1], cost_data, label='Cost', marker='+', linestyle='None', markersize=4)
plt.title('cost function')
plt.figure(2)
plt.plot(tspan[:-1], time_data, label='CPU Time', marker='+', linestyle='None', markersize=4)
plt.title('cpu time')


# # build the plots
# print(tspan.shape, state_data.shape)
plot_state(tspan, residual_state, 'state residuals')
plot_control(tspan, residual_control, 'control residuals')
plot_state(tspan, state_data, 'state')
plot_state(tspan[:-1], error, 'state error')
# plot_state(tspan[:-1], cum_error, 'state residuals')
plot_control(tspan, control_data, 'control flight data')
plot_control(tspan[:-1], control_data_computed, 'control computed')
plot_pwm(tspan, pwm_servos, pwm_motors, 'pwm')

plt.figure()
plt.plot(tspan, voltage)
plt.title('voltage')

plt.show()





