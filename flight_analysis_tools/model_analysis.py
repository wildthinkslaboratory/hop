# Analysis tools for flight logs
#
from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
from hop.utilities import quaternion_to_angle
from flight_analysis_tools.flight_data import FlightData
import casadi as ca
import numpy as np
import statistics as stats
from time import perf_counter
import matplotlib.pyplot as plt
from plotting.plots import plot_state, plot_control, plot_pwm, plot_attitude, plot_parameters, plot_weighted_error_state, plot_weighted_error_control
from hop.equations_of_motion import Equations6DOF


mc = Constants()
equations = Equations6DOF(mc)
fd = FlightData()

# update the constants with those used in the flight
mc.update_from_dictionary(fd.constants)

model = DroneModel(mc)  

# create an nmpc to compute the control
# We run the nmpc on the flight state and see if the computed control
# matches the flight control for each timestep
mpc = DroneNMPCdompc(mc.dt, model.model)
mpc.setup_cost()

# data structures for the things we want to plot
tspan = np.arange(0, fd.len_used_data * fd.dt , fd.dt)
flight_model_error = np.empty([fd.len_used_data-1,13])
control_data_computed = np.empty([fd.len_used_data-1,4])
control_computed_diff = np.empty([fd.len_used_data-1,4])
predicted_state = np.empty([fd.len_used_data-1,13])
residual_state = np.empty([fd.len_used_data,13])
residual_control = np.empty([fd.len_used_data,4])
attitude = np.empty([fd.len_used_data,3])
timing_int = np.empty([fd.len_used_data,2])
time_data = []
cost_data = []

# set the start times for pi and px4
px4_start_time = fd.timing_data[0][0]
pi_start_time = fd.timing_data[0][2]

# Now we set the initial state
x_init = ca.DM(fd.state_data[0])
xr = mc.xr
mpc.set_start_state(x_init)
x0 = x_init

# these should be read in from logged constants
xrnp = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
urnp = np.array([0.0, 0.0, mc.hover_thrust, 0.0])

# run the simulation
took_too_long = 0
for i in range(len(fd.state_data)-1):

    if i > 0:
        tstep = fd.timestamps[i] - fd.timestamps[i-1]
        if tstep > 0.025:
            print(i, tstep)
            
    # update parameters with current voltage
    fd.parameters[i][3] = fd.voltage[i]

    # first run the nmpc on the state
    start_time = perf_counter()

    mpc.set_waypoint(np.array(fd.parameters[i]))
    u0 = mpc.mpc.make_step(fd.state_data[i])
    step_time = perf_counter() - start_time
    

    # we want to check the nmpc results for the flight
    control_data_computed[i] = np.reshape(u0, (4,))
    control_computed_diff[i] = fd.control_data[i] - control_data_computed[i]
    time_data.append(step_time)
    cost_data.append(mpc.mpc.data['_aux'][-1][2])
    if not mpc.mpc.solver_stats['return_status'] == 'Solve_Succeeded':
        print(mpc.mpc.solver_stats['return_status'])


    # turn quaternions into attitude
    # it's easier to read
    q = fd.state_data[6:10].copy()
    q = np.reshape(fd.state_data[i][6:10].copy(), (4,))
    attitude[i] = quaternion_to_angle(q)
    

    # compute the weighted squared error
    state_error = fd.state_data[i] - xrnp
    control_error = fd.control_data[i] - urnp
    for j in range(len(state_error)):
        residual_state[i][j] = state_error[j] * mc.Q[j,j] * state_error[j]
    residual_control[i] = np.absolute(control_error)
    for j in range(len(control_error)):
        residual_control[i][j] = control_error[j] * mc.R[j,j] * control_error[j]

    # predict the next state and compare with actual next state
    x0 = fd.state_data[i] + mc.dt* equations.f(fd.state_data[i],np.reshape(fd.control_data[i], (4,1)), fd.parameters[i])
    flight_model_error[i] = fd.state_data[i+1] -  np.reshape(x0, (13,))
    predicted_state[i] = np.reshape(x0, (13,))
    timing_int[i] = np.array([fd.timing_data[i][0]-px4_start_time, 0.0])



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

plt.figure(3)
plt.plot(tspan, fd.voltage)
plt.title('voltage')

plot_parameters(tspan[:-1], fd.parameters, 'parameters')
plot_pwm(tspan, fd.pwm_servos, fd.pwm_motors, 'pwm')

plot_control(tspan[:-1], control_data_computed, 'control computed')
plot_control(tspan[:-1], control_computed_diff, 'control computed difference')
plot_state(tspan[:-1], flight_model_error, 'flight state vs model predicted state error')
plot_weighted_error_state(tspan, residual_state, 'state weighted squared errors')
plot_weighted_error_control(tspan, residual_control, 'control weighted squared errors')
plot_attitude(tspan, attitude, 'attitude')
plot_control(tspan, fd.control_data, 'flight control data')
plot_state(tspan, fd.state_data, 'state')

plt.show()





