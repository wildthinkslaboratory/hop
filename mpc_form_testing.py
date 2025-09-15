#
# This runs drone simulations, plots results and gives timing summaries
#
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


# If you just want to run a single test you can loop over this list
test_list = [
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

# Here is the full set of tests if you want to run all the simulations
test_list_full = [
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

for test in test_list:

    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)


    # first we set up the do-mpc solver
    # it uses orthagonal collocation
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

    # run do-mpc solver
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
        

    # set up the Chebyshev pseudospectral nmpc solver
    cheb_nmpc = DroneNMPCwithCGL()
    cheb_nmpc.set_goal_state(xr)
    cheb_nmpc.set_start_state(x_init)
    cheb_nmpc_state_data = np.empty([num_iterations,13])
    cheb_nmpc_control_data = np.empty([num_iterations,4])
    cheb_nmpc_time_data = []
    x0 = x_init
    u0 = np.zeros(4)


    print('running Chebyshev pseudospectral nmpc solver')
    for k in range(num_iterations):

        start_time = perf_counter()
        # Solve the NMPC for the current state x_current
        u0 = cheb_nmpc.make_step(x0, u0)
        step_time = perf_counter() - start_time

        # Propagate the system using the discrete dynamics f (Euler forward integration)
        x0 = x0 + mc.dt* cheb_nmpc.f(x0,u0)
        
        cheb_nmpc_state_data[k] = np.reshape(x0, (13,))
        cheb_nmpc_control_data[k] = np.reshape(u0, (4,))
        cheb_nmpc_time_data.append(step_time)


    # run the multiple shooter nmpc
    ms_nmpc = DroneNMPCSingleShoot()
    ms_nmpc.set_goal_state(xr)
    ms_nmpc.set_start_state(x_init)
    ms_nmpc_state_data = np.empty([num_iterations,13])
    ms_nmpc_control_data = np.empty([num_iterations,4])
    ms_nmpc_time_data = []
    x0 = x_init

    print('running multiple shooter nmpc solver')
    for k in range(num_iterations):

        start_time = perf_counter()
        # Solve the NMPC for the current state x_current
        u0 = ms_nmpc.make_step(x0, u0)
        step_time = perf_counter() - start_time

        # Propagate the system using the discrete dynamics f (Euler forward integration)
        x0 = x0 + mc.dt* ms_nmpc.f(x0,u0)
        
        ms_nmpc_state_data[k] = np.reshape(x0, (13,))
        ms_nmpc_control_data[k] = np.reshape(u0, (4,))
        ms_nmpc_time_data.append(step_time)

    # compute statistics for the timing of the nmpc calls
    mean_time = [round(t,3) for t in [stats.mean(dompc_time_data), stats.mean(cheb_nmpc_time_data), stats.mean(ms_nmpc_time_data)]]
    max_time = [round(t,3) for t in [max(dompc_time_data), max(cheb_nmpc_time_data),  max(ms_nmpc_time_data)]]
    bad_times = [len([b for b in dompc_time_data if b > 0.014]), 
                 len([b for b in cheb_nmpc_time_data if b > 0.014]), 
                 len([b for b in ms_nmpc_time_data if b > 0.014])]
    
    # print timing results
    print(test['title'])
    print("          {: >20} {: >20} {: >20}".format('do-mpc', 'pseudospectral', 'multiple shoot'))
    print("-----------------------------------------------------------------------------------------")
    print("mean      {: >20} {: >20} {: >20}".format(*mean_time))
    print("max       {: >20} {: >20} {: >20}".format(*max_time))
    print("bad times {: >20} {: >20} {: >20}".format(*bad_times))


    plot_state_for_paper(tspan, dompc_state_data, test["title"], 1)
    plot_state_for_paper(tspan, cheb_nmpc_state_data, test["title"], 2)
    plot_state_for_paper(tspan, ms_nmpc_state_data, test["title"], 3)


    plot_control_for_paper(tspan, dompc_control_data, test["title"], 4)
    plot_control_for_paper(tspan, cheb_nmpc_control_data, test["title"], 5)
    plot_control_for_paper(tspan, ms_nmpc_control_data, test["title"], 6)

    plot_time_comparison(tspan, dompc_time_data, ms_nmpc_time_data, cheb_nmpc_time_data, test["title"], 7)
    plt.show()



    