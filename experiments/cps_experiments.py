#

# This is a set of experiments for tuning the chebyshev pseudo-spectral implementation
#


from hop.constants import Constants


import casadi as ca
import numpy as np
import statistics as stats
from hop.utilities import import_data
from time import perf_counter
from hop.chebyshev_ps import DroneNMPCwithCPS
import matplotlib.pyplot as plt
from simulation_tools.integrators import RKSimulator
from plotting.plots import plot_comparison, plot_state_for_paper, plot_control_for_paper

mc = Constants()

# If you just want to run a single test you can loop over this list
single_test = [
{
    "x0": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [1, -0.5, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 250,
    "waypoint": [0.0, 0.0, 0.0],
    "title": "x1z1"
  },
]


# Here is the full set of tests if you want to run all the simulations
# test_list_for_paper = import_data('nmpc_test_cases.json')
test_list_for_paper = single_test

spectral_order = [4, 6, 8, 10]

for test in test_list_for_paper:
    # print timing results
    print(test['title'])
    s = ["{: >20} ".format(p) for p in ['N', 'time', 'settle']]
    print(''.join(s))
    print("-----------------------------------------------------------------------------------------")    
    for order in spectral_order:

        # set up the test case
        num_iterations = test['num_iterations']
        x_init = ca.DM(test['x0'])
        xr = np.array(test['xr'])
        allowed_error = np.array([0.05,0.05,0.05, 0.02,0.02,0.02, 0.02,0.02,0.02,0.02, 0.01,0.01,0.01])
        goal_ul = xr + allowed_error
        goal_ll = xr - allowed_error
        tspan = np.arange(0,num_iterations* mc.dt,mc.dt)
        horizon_time = 1.0

        state_data = np.empty([num_iterations,13])
        control_data = np.empty([num_iterations,4])
        time_data = []
        cost_data = []
        stats_data = 0

        # set up the Chebyshev pseudospectral nmpc solver
        cheb_nmpc = DroneNMPCwithCPS(mc)
        cheb_nmpc.N = order
        cheb_nmpc.T = horizon_time
        cheb_nmpc.record_nlp_stats = True

        cheb_nmpc.build_nmpc_instance()
        cheb_nmpc.set_start_state(x_init)
        x0 = x_init
        u0 = np.zeros(4)
        rk_sim = RKSimulator(0.005, 4)
        params = np.array([xr[0], xr[1], xr[2], mc.battery_v, mc.hover_thrust])


        for k in range(num_iterations):

            start_time = perf_counter()
            # Solve the NMPC for the current state x_current
            u0 = cheb_nmpc.make_step(x0, u0, params)
            step_time = perf_counter() - start_time

            # Propagate the system using the discrete dynamics f
            # runge kutta 4 simulator
            x0 = rk_sim.make_step(cheb_nmpc.f, x0, u0, params)  

            
            state_data[k] = np.reshape(x0, (13,))
            control_data[k] = np.reshape(u0, (4,))
            time_data.append(step_time)
            cost_data.append(cheb_nmpc.solver_stats['cost'])
            if not cheb_nmpc.solver_stats['status'] == 'Solve_Succeeded':
                stats_data += 1
                print(cheb_nmpc.solver_stats['status'])



        from experiments.trajectory_metrics import settling_metric
        settle = round(settling_metric(state_data, goal_ll, goal_ul) * mc.dt, 3) 
        mean_time = round(stats.mean(time_data),3)
        s = ["{: >20} ".format(v) for v in [cheb_nmpc.N, mean_time, settle]]
        print(''.join(s))


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


