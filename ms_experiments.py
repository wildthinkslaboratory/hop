#
# This is for tuning the multipleShooting implementation
#

from hop.constants import Constants

import casadi as ca
import numpy as np
import statistics as stats
from hop.utilities import import_data, sig_figs
from time import perf_counter
from hop.multiShooting import DroneNMPCMultiShoot
import matplotlib.pyplot as plt
from plots import plot_comparison, plot_state_for_paper, plot_control_for_paper

mc = Constants()

# If you just want to run a single test you can loop over this list
single_test = [
  {
    "x0": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-1, -0.1, -0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 250,
    "title": "y115dx"
  },
]


# Here is the full set of tests if you want to run all the simulations
test_list_for_paper = import_data('nmpc_test_cases.json')

time_steps = [0.02, 0.025, 0.04, 0.05, 0.1, 0.2, 0.4, 0.5, 1]
# time_steps = [0.4, 0.5, 1]

for test in test_list_for_paper:
    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # run fine grained solver for a reference trajectory
    # the accuracy of other runs are assessed relative to this trajectory
    ms_nmpc = DroneNMPCMultiShoot()
    ms_nmpc.dt = 0.02
    ms_nmpc.N = 100
    ms_nmpc.record_nlp_stats = True
    ms_nmpc.set_goal_state(xr)
    ms_nmpc.set_start_state(x_init)
    x0 = x_init
    u0 = np.zeros(4)
    reference_data = np.empty([num_iterations,13])

    print('running multiple shooter nmpc solver with N:', ms_nmpc.N)
    for k in range(num_iterations):

        u0 = ms_nmpc.make_step(x0, u0, np.array([0.0, 0.0, 0.0]))
        x0 = x0 + mc.dt* ms_nmpc.f(x0,u0)
        reference_data[k] = np.reshape(x0, (13,))


    for ts in time_steps:

        state_data = np.empty([num_iterations,13])
        control_data = np.empty([num_iterations,4])
        time_data = []
        cost_data = []
        stats_data = 0


        # run the multiple shooter nmpc
        ms_nmpc = DroneNMPCMultiShoot()
        ms_nmpc.dt = ts
        ms_nmpc.N = int(2.0 / ts)
        ms_nmpc.record_nlp_stats = True
        ms_nmpc.set_goal_state(xr)
        ms_nmpc.set_start_state(x_init)
        x0 = x_init
        u0 = np.zeros(4)

        print('running multiple shooter nmpc solver with N:', ms_nmpc.N)
        for k in range(num_iterations):

            start_time = perf_counter()
            # Solve the NMPC for the current state x_current
            u0 = ms_nmpc.make_step(x0, u0, np.array([0.0, 0.0, 0.0]))
            step_time = perf_counter() - start_time

            # Propagate the system using the discrete dynamics f (Euler forward integration)
            x0 = x0 + mc.dt* ms_nmpc.f(x0,u0)
            
            state_data[k] = np.reshape(x0, (13,))
            control_data[k] = np.reshape(u0, (4,))
            time_data.append(step_time)
            cost_data.append(ms_nmpc.solver_stats['cost'])
            if not ms_nmpc.solver_stats['status'] == 'Solve_Succeeded':
                stats_data += 1

                    # compute the accuracy metric
        state_error = 0
        for i in range(num_iterations):
            error = reference_data[i] - state_data[i]
            state_error += error.T @ error

        accuracy = sig_figs(state_error, 2)

        print(test['title'])
        print("Horizon: ", ms_nmpc.N)
        print('mean time: ', round(stats.mean(time_data),3))
        print('max time:  ', round(max(time_data),3))
        print('fails:     ', stats_data)
        print('accuracey: ', accuracy)

        plot_state_for_paper(tspan, state_data, test["title"], 1)
    

        plot_control_for_paper(tspan, control_data, test["title"], 2)

        times = {'ms': time_data}
        costs = {'ms': cost_data}
        plot_comparison(tspan, times, test["title"], 3, 'CPU Time (sec)')
        plot_comparison(tspan, costs, test["title"], 4, 'Cost')

        plt.show()


