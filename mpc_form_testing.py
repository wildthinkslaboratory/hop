from hop.drone_model import DroneModel
from hop.drone_mpc import DroneMPC
from hop.constants import Constants
from do_mpc.simulator import Simulator
import casadi as ca
from do_mpc.estimator import StateFeedback
import numpy as np
import statistics as stats
from hop.utilities import import_data
from time import perf_counter
from hop.drone_mpc_casadi import DroneNMPCSingleShoot
from hop.drone_mpc_cgl import DroneNMPCwithCGL
from animation import RocketAnimation
import matplotlib.pyplot as plt
from plots import plot_state_for_comparison, plot_control_for_comparison, plot_time_comparison, plot_state_for_paper, plot_control_for_paper

mc = Constants()

tests = [
  {
    "x0": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-1, -0.1, -0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 250,
    "title": "y115dxT"
  },
]

tests_full = [
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [0.0, -0.2, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.8,
    "num_iterations": 200,
    "title": "hover"
  },
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.383, 0.924, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-0.2, -0.5, 0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.8,
    "num_iterations": 500,
    "title": "45dz"
  },
  {
    "x0": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [0.0, -0.2, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.8,
    "num_iterations": 400,
    "title": "z1"
  },
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [0.0, -0.2, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.8,
    "num_iterations": 200,
    "title": "y1"
  },
  {
    "x0": [1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [1, -0.5, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 200,
    "title": "x1z1vx"
  },
  {
    "x0": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-1, -0.1, -0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 250,
    "title": "y115dx"
  },
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-1, -0.1, -0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 250,
    "title": "starting 15 deg around x"
  },
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
        

    specmpc = DroneNMPCwithCGL()
    specmpc.set_goal_state(xr)
    specmpc.set_start_state(x_init)
    specmpc_state_data = np.empty([num_iterations,13])
    specmpc_control_data = np.empty([num_iterations,4])
    specmpc_time_data = []
    x0 = x_init
    u0 = np.zeros(4)
    # u0 = np.array([0.0, 0.0, 5.67, 0.0])

    print('running spectral mpc solver')
    for k in range(num_iterations):

        start_time = perf_counter()
        # Solve the NMPC for the current state x_current
        u0 = specmpc.make_step(x0, u0)
        step_time = perf_counter() - start_time

        # Propagate the system using the discrete dynamics f (Euler forward integration)
        x0 = x0 + mc.dt* specmpc.f(x0,u0)
        
        specmpc_state_data[k] = np.reshape(x0, (13,))
        specmpc_control_data[k] = np.reshape(u0, (4,))
        specmpc_time_data.append(step_time)

    ssmpc = DroneNMPCSingleShoot()
    ssmpc.set_goal_state(xr)
    ssmpc.set_start_state(x_init)
    ssmpc_state_data = np.empty([num_iterations,13])
    ssmpc_control_data = np.empty([num_iterations,4])
    ssmpc_time_data = []
    x0 = x_init

    print('running single shoot mpc solver')
    for k in range(num_iterations):

        start_time = perf_counter()
        # Solve the NMPC for the current state x_current
        u0 = ssmpc.make_step(x0, u0)
        step_time = perf_counter() - start_time

        # Propagate the system using the discrete dynamics f (Euler forward integration)
        x0 = x0 + mc.dt* ssmpc.f(x0,u0)
        
        ssmpc_state_data[k] = np.reshape(x0, (13,))
        ssmpc_control_data[k] = np.reshape(u0, (4,))
        ssmpc_time_data.append(step_time)

    mean_time = [round(t,3) for t in [stats.mean(dompc_time_data), stats.mean(specmpc_time_data), stats.mean(ssmpc_time_data)]]
    max_time = [round(t,3) for t in [max(dompc_time_data), max(specmpc_time_data),  max(ssmpc_time_data)]]
    bad_times = [len([b for b in dompc_time_data if b > 0.014]), 
                 len([b for b in specmpc_time_data if b > 0.014]), 
                 len([b for b in ssmpc_time_data if b > 0.014])]
    
    print(test['title'])
    print("          {: >20} {: >20} {: >20}".format('do-mpc', 'spectral', 'sing shoot'))
    print("-----------------------------------------------------------------------------------------")
    print("mean      {: >20} {: >20} {: >20}".format(*mean_time))
    print("max       {: >20} {: >20} {: >20}".format(*max_time))
    print("bad times {: >20} {: >20} {: >20}".format(*bad_times))


    plot_state_for_paper(tspan, dompc_state_data, test["title"], 1)
    plot_state_for_paper(tspan, specmpc_state_data, test["title"], 2)
    plot_state_for_paper(tspan, ssmpc_state_data, test["title"], 3)

    plot_control_for_paper(tspan, dompc_control_data, test["title"], 4)
    plot_control_for_paper(tspan, specmpc_control_data, test["title"], 5)
    plot_control_for_paper(tspan, ssmpc_control_data, test["title"], 6)

    # plot_state_for_comparison(tspan, dompc_state_data, test["title"], 1)
    # plot_state_for_comparison(tspan, specmpc_state_data, test["title"], 2)
    # plot_state_for_comparison(tspan, ssmpc_state_data, test["title"], 3)

    # plot_control_for_comparison(tspan, dompc_control_data, test["title"], 4)
    # plot_control_for_comparison(tspan, specmpc_control_data, test["title"], 5)
    # plot_control_for_comparison(tspan, ssmpc_control_data, test["title"], 6)

    plot_time_comparison(tspan, dompc_time_data, ssmpc_time_data, specmpc_time_data, test["title"], 7)
    plt.show()

    # input("Press [enter] to continue.")
    # rc = RocketAnimation(test['animation_forward'], test['animation_up'], test['animation_frame_rate'])
    # rc.animate(tspan, specmpc_state_data, specmpc_control_data)



    