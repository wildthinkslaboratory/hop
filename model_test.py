import matplotlib.pyplot as plt
from hop.drone_model import DroneModel
from hop.drone_mpc import DroneMPC
from hop.constants import Constants
from do_mpc.simulator import Simulator
from do_mpc import graphics
import casadi as ca
from do_mpc.estimator import StateFeedback
import numpy as np
from hop.utilities import import_data
from time import perf_counter
from animation import RocketAnimation
from plots import plot_state, plot_control

mc = Constants()

num_iterations = 200
plot = True
run_animation = True

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
    state_data = np.empty([num_iterations,13])
    control_data = np.empty([num_iterations,4])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)
    time_data = []
    for k in range(num_iterations):
        start_time = perf_counter()
        u0 = mpc.mpc.make_step(x0)
        step_time = perf_counter() - start_time

        y_next = sim.make_step(u0)
        x0 = estimator.make_step(y_next)


        state_data[k] = np.reshape(x0, (13,))
        # u0[0] = u0[0] * 180 / np.pi
        # u0[1] = u0[1] * 180 / np.pi
        control_data[k] = np.reshape(u0, (4,))
        time_data.append(step_time)

    # now we analyze the data
    cum_time = 0.0

    for t in time_data:
        cum_time += t
        if (t > mc.dt):
            print('mpc timestep exceeded:',t)
    print('average time for mpc step: ', cum_time / num_iterations)

    plot_state(tspan, state_data)
    plot_control(tspan, control_data)
    
    if run_animation:
        rc = RocketAnimation()
        rc.animate(tspan, state_data, control_data)