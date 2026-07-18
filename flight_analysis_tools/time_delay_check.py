# This is the code we used to find the time delay.
#

from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc
from flight_analysis_tools.flight_data import FlightData
# from hop.multiShooting import DroneNMPCMultiShoot
from hop.constants import Constants
from hop.equations_of_motion import Equations6DOF
from hop.utilities import  import_data
import casadi as ca
import numpy as np

import matplotlib.pyplot as plt
from plotting.plots import plot_state, plot_control


# first we make a model
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


# ms_mpc = DroneNMPCMultiShoot(mc)

# data structures for the things we want to plot
tspan = np.arange(0, fd.len_used_data * fd.dt , fd.dt)
flight_model_error = np.empty([fd.len_used_data-1,13])
dx_model_error = np.empty([fd.len_used_data-1,13])
wx_data = np.empty([fd.len_used_data-1,6])
state_change = np.empty([fd.len_used_data-1,13])
model_change_state = np.empty([fd.len_used_data-1,13])

fd.control_data_computed = np.empty([fd.len_used_data-1,4])
control_computed_diff = np.empty([fd.len_used_data-1,4])
predicted_state = np.empty([fd.len_used_data-1,13])
residual_state = np.empty([fd.len_used_data,13])
residual_control = np.empty([fd.len_used_data,4])
attitude = np.empty([fd.len_used_data,3])
time_data = []
cost_data = []


# Now we set the initial state
x_init = ca.DM(fd.state_data[0])
xr = mc.xr
mpc.set_start_state(x_init)
x0 = x_init

# these should be read in from logged constants
xrnp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
urnp = np.array([0.0, 0.0, mc.hover_thrust, 0.0])
delay_steps = 0
thrust_delay = 0
servo_delay = 0

# run the simulation
for i in range(delay_steps,len(fd.state_data)-2):

    # update fd.parameters with current fd.voltage
    fd.parameters[i][3] = fd.voltage[i]

   # predict the next state and compare with actual next state
    u_delayed = np.array([fd.control_data[i-servo_delay][0], 
                          fd.control_data[i-servo_delay][1],
                          fd.control_data[i-thrust_delay][2],
                          fd.control_data[i-thrust_delay][3]
                          ])
    # dx = ms_mpc.f(fd.state_data[i],u_delayed, fd.parameters[i])
    dx = equations.f(fd.state_data[i],u_delayed, fd.parameters[i])
    

    x0 = fd.state_data[i] + mc.dt* dx
    dx = np.reshape(dx, (13,))

    flight_model_error[i] = fd.state_data[i+1] -  np.reshape(x0, (13,))
    dx_model_error[i] = (fd.state_data[i+1] - fd.state_data[i]) -  np.reshape(dx, (13,))

    state_change[i] = fd.state_data[i+1] - fd.state_data[i]
    model_change_state[i] = dx * mc.dt

    predicted_state[i] = np.reshape(x0, (13,))


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
plot_control(tspan, fd.control_data, 'flight control data')

plot_w_dx(tspan[:-1], model_change_state, state_change)
plot_v_dx(tspan[:-1], model_change_state, state_change)
plt.show()



