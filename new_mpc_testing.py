from hop.drone_model import DroneModel
from hop.drone_mpc import DroneMPC
from hop.constants import Constants
from do_mpc.simulator import Simulator
import casadi as ca
from do_mpc.estimator import StateFeedback
import numpy as np
from hop.utilities import import_data
from time import perf_counter
from hop.drone_mpc_casadi import DroneNMPCCasadi
from animation import RocketAnimation
import matplotlib.pyplot as plt
from plots import plot_state_for_comparison, plot_control_for_comparison

mc = Constants()

tests = [
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
    "xr": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-1, -0.1, -0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 250,
    "title": "hop in y direction, starting 15 deg around x"
  }
]


for test in tests:

    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # first we set up the old nmpc solver
    model = DroneModel()
    mpc = DroneMPC(mc.dt, model.model)
    estimator = StateFeedback(model.model)
    sim = Simulator(model.model)
    sim.set_param(t_step = mc.dt)
    sim.setup()
    mpc.set_goal_state(xr)
    mpc.set_start_state(x_init)
    sim.x0 = x_init
    estimator.x0 = x_init
    dompc_state_data = np.empty([num_iterations,13])
    dompc_control_data = np.empty([num_iterations,4])
    dompc_time_data = []
    x0 = x_init
    # second set up the new nmpc solver

    # run old solver
    print('running do-mpc solver')
    for k in range(num_iterations):
        start_time = perf_counter()
        u0 = mpc.mpc.make_step(x0)
        step_time = perf_counter() - start_time

        y_next = sim.make_step(u0)
        x0 = estimator.make_step(y_next)

        dompc_state_data[k] = np.reshape(x0, (13,))
        dompc_control_data[k] = np.reshape(u0, (4,))
        dompc_time_data.append(step_time)
        

    newmpc = DroneNMPCCasadi()
    newmpc.set_goal_state(xr)
    newmpc.set_start_state(x_init)
    newmpc_state_data = np.empty([num_iterations,13])
    newmpc_control_data = np.empty([num_iterations,4])
    newmpc_time_data = []
    x0 = x_init

    print('running new mpc solver')
    for k in range(num_iterations):

        # Solve the NMPC for the current state x_current
        u0 = newmpc.make_step(x0)
        
        # Propagate the system using the discrete dynamics f (Euler forward integration)
        x0 = x0 + mc.dt* newmpc.f(x0,u0)
        
        newmpc_state_data[k] = np.reshape(x0, (13,))
        newmpc_control_data[k] = np.reshape(u0, (4,))




    plot_state_for_comparison(tspan, dompc_state_data, test["title"] + ' dompc', 1)
    plot_state_for_comparison(tspan, newmpc_state_data, test["title"] + ' newmpc', 2)

    plot_control_for_comparison(tspan, dompc_control_data, test["title"] + ' dompc', 3)
    plot_control_for_comparison(tspan, newmpc_control_data, test["title"] + ' newmpc', 4)

    plt.show()

    # input("Press [enter] to continue.")
    # rc = RocketAnimation(test['animation_forward'], test['animation_up'], test['animation_frame_rate'])
    # rc.animate(tspan, newmpc_state_data, newmpc_control_data)



    