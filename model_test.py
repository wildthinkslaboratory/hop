import matplotlib.pyplot as plt
from drone_model import DroneModel
from drone_mpc import DroneMPC
from constants import Constants
from do_mpc.simulator import Simulator
from do_mpc import graphics
import casadi as ca
from do_mpc.estimator import StateFeedback
import numpy as np
from utilities import import_data
from time import perf_counter
from animation import RocketAnimation

mc = Constants()

num_iterations = 200
plot = True
run_animation = False

if run_animation:
    rc = RocketAnimation()

tests = import_data('nmpc_test_cases.json')
for test in tests:
    model = DroneModel()
    mpc = DroneMPC(mc.dt, model.model)

    estimator = StateFeedback(model.model)
    sim = Simulator(model.model)
    sim.set_param(t_step = mc.dt)
    sim.setup()

    # initial state
    x0 = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    mpc.set_goal_state(xr)
    mpc.set_start_state(x0)
    sim.x0 = x0
    estimator.x0 = x0
    data = []
    state_data = np.empty([num_iterations,12])
    control_data = np.empty([num_iterations,4])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)
    time_data = []
    for k in range(num_iterations):
        start_time = perf_counter()
        u0 = mpc.mpc.make_step(x0)
        step_time = perf_counter() - start_time

        y_next = sim.make_step(u0)
        x0 = estimator.make_step(y_next)


        state_data[k] = np.reshape(x0, (12,))
        control_data[k] = np.reshape(u0, (4,))
        time_data.append(step_time)

    # now we analyze the data
    cum_time = 0.0

    for t in time_data:
        cum_time += t
        if (t > mc.dt):
            print('mpc timestep exceeded:',t)
    print('average time for mpc step: ', cum_time / num_iterations)


    if plot:
        fig, axs = plt.subplots(6)
        fig.set_figheight(8)
        fig.suptitle("NMPC Drone Simulation")

        for i in range(3):
            axs[0].plot(tspan, state_data[:,i])
        axs[0].set_ylabel('x')
        for i in range(3):
            axs[1].plot(tspan, state_data[:,i+3])
        axs[1].set_ylabel('v')
        for i in range(3):
            axs[2].plot(tspan, state_data[:,i+6])
        axs[2].set_ylabel('q')
        for i in range(3):
            axs[3].plot(tspan, state_data[:,i+9])
        axs[3].set_ylabel('w')

        for i in range(2):
            axs[4].plot(tspan, control_data[:,i])
        axs[4].set_ylabel('g')
        for i in range(2):
            axs[5].plot(tspan, control_data[:,i+2])
        axs[5].set_ylabel('t')

        plt.xlabel('Time')
        plt.show()
    
    if run_animation:
        rc.animate(tspan, state_data, control_data)