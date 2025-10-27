#
# This runs drone simulations, plots results and gives timing summaries
#
from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
from do_mpc.simulator import Simulator
import casadi as ca
from do_mpc.estimator import StateFeedback
import numpy as np
import statistics as stats
from hop.utilities import import_data
from time import perf_counter
from hop.multiShooting import DroneNMPCMultiShoot
from hop.chebyshev_ps import DroneNMPCwithCPS
from animation import RocketAnimation
import matplotlib.pyplot as plt
from plots import plot_comparison, plot_state_for_paper, plot_control_for_paper

mc = Constants()

# If you just want to run a single test you can loop over this list
single_test = [
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [0.0, -0.2, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.8,
    "num_iterations": 200,
    "title": "hover"
  },
]


# Here is the full set of tests if you want to run all the simulations
test_list_for_paper = import_data('nmpc_test_cases.json')

spectral_order = [16, 20]
# spectral_order = [6, 8, 10, 12, 14]
for order in spectral_order:
    for test in test_list_for_paper:

        # set up the test case
        num_iterations = test['num_iterations']
        x_init = ca.DM(test['x0'])
        xr = ca.DM(test['xr'])
        tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

        state_data = np.empty([num_iterations,13])
        control_data = np.empty([num_iterations,4])
        time_data = []
        cost_data = []
        stats_data = 0

        # set up the Chebyshev pseudospectral nmpc solver
        cheb_nmpc = DroneNMPCwithCPS()
        cheb_nmpc.N = order
        cheb_nmpc.record_nlp_stats = True
        cheb_nmpc.set_goal_state(xr)
        cheb_nmpc.set_start_state(x_init)
        x0 = x_init
        u0 = np.zeros(4)


        print('running Chebyshev pseudospectral nmpc solver')
        for k in range(num_iterations):

            start_time = perf_counter()
            # Solve the NMPC for the current state x_current
            u0 = cheb_nmpc.make_step(x0, u0, np.array([0.0, 0.0, 0.0]))
            step_time = perf_counter() - start_time

            # Propagate the system using the discrete dynamics f (Euler forward integration)
            x0 = x0 + mc.dt* cheb_nmpc.f(x0,u0)
            
            state_data[k] = np.reshape(x0, (13,))
            control_data[k] = np.reshape(u0, (4,))
            time_data.append(step_time)
            cost_data.append(cheb_nmpc.solver_stats['cost'])
            if not cheb_nmpc.solver_stats['status'] == 'Solve_Succeeded':
                stats_data += 1



        print(test['title'])
        print("Spectral Order: ", cheb_nmpc.N)
        print('mean time: ', round(stats.mean(time_data),3))
        print('max time:  ', round(max(time_data),3))
        print('fails:     ', stats_data)

        plot_state_for_paper(tspan, state_data, test["title"], 1)
        # plt.savefig("cpsState.pdf", format="pdf", bbox_inches="tight")
        plot_control_for_paper(tspan, control_data, test["title"], 2)
        # plt.savefig("cpsControl.pdf", format="pdf", bbox_inches="tight")
        times = {'ms': time_data}
        costs = {'ms': cost_data}
        plot_comparison(tspan, times, test["title"], 3, 'CPU Time (sec)')
        # plt.savefig("cpsTime.pdf", format="pdf", bbox_inches="tight")
        plot_comparison(tspan, costs, test["title"], 4, 'Cost')
        # plt.savefig("cpsCost.pdf", format="pdf", bbox_inches="tight")

        plt.show()


