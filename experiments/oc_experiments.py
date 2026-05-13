#
# This runs experiments on do-mpc's orthogonal collocation method to 
# determine the optimal number of intervals and the number of collocation points.
# This should produce two 2D heat maps one for accuracy and one for time
# with the 2D grid corresponding to combinations of number of collocation points
# and size of intervals
#
from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
import casadi as ca
import numpy as np
import statistics as stats
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib import colors
from plotting.plots import plot_comparison, plot_state_for_paper, plot_control_for_paper
from hop.utilities import sig_figs
from hop.multiShooting import DroneNMPCMultiShoot
from simulation_tools.integrators import RKSimulator
mc = Constants()


# # If you just want to run a single test you can loop over this list
# test_list = [
#   {
#     "x0": [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
#     "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#     "animation_forward": [-1, -0.1, -0.2],
#     "animation_up": [0, 1, 0],
#     "animation_frame_rate": 0.4,
#     "num_iterations": 200,
#     "title": "x1y1z05_15deg"
#   }
# ]

test_list = [
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


collocations = [1, 2]                        # number of collocation points
timesteps = [0.2, 0.25, 0.3]   # size of the time intervals (within a 2 second horizon)

times = np.zeros((len(collocations),len(timesteps)))     # we measure the average solution time
accuracy = np.zeros((len(collocations),len(timesteps)))  # we measure the accuracy


for test in test_list:
    
    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = np.array(test['xr'])
    allowed_error = np.array([0.05,0.05,0.05, 0.02,0.02,0.02, 0.02,0.02,0.02,0.02, 0.01,0.01,0.01])
    goal_ul = xr + allowed_error
    goal_ll = xr - allowed_error
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # set up the Runge-Kutta simulator
    ms_model = DroneNMPCMultiShoot(mc)
    rk_sim = RKSimulator(0.005, 4)
    params = np.array([xr[0], xr[1], xr[2], mc.battery_v, mc.hover_thrust])

    # run fine grained solver for a reference trajectory
    # the accuracy of other runs are assessed relative to this trajectory
    model = DroneModel(mc)
    mpc = DroneNMPCdompc(mc.dt, model.model)

    horizon_time = 1.2
    mpc.mpc.settings.t_step = 0.02
    mpc.mpc.settings.n_horizon = int(horizon_time / 0.02)
    mpc.mpc.settings.collocation_deg = 3 


    mpc.setup_cost()
    mpc.set_start_state(x_init)
    reference_data = np.empty([num_iterations,13])
    x0 = x_init

    print('running reference do-mpc solver')
    for k in range(num_iterations):
        start_time = perf_counter()
        mpc.set_waypoint(params)
        u0 = mpc.mpc.make_step(x0)
        step_time = perf_counter() - start_time

        # runge kutta 4 simulator
        x0 = rk_sim.make_step(ms_model.f, x0, u0, params)
        reference_data[k] = np.reshape(x0, (13,))

                
    # print timing results
    print(test['title'])
    s = ["{: >20} ".format(p) for p in ['step', 'deg', 'time', 'score', 'settle']]
    print(''.join(s))
    print("-----------------------------------------------------------------------------------------")

    # These are our experimental runs 
    # loop over the number of collocation and 
    # size of the time intervals
    for c in collocations:
        for n,tstep in enumerate(timesteps):
            n_horizon = int(horizon_time / tstep)

            # first we set up the do-mpc solver
            # it uses orthagonal collocation
            model = DroneModel(mc)
            mpc = DroneNMPCdompc(mc.dt, model.model)

            mpc.mpc.settings.t_step = tstep
            mpc.mpc.settings.n_horizon = n_horizon
            mpc.mpc.settings.collocation_deg = c  

            mpc.setup_cost()
            mpc.set_start_state(x_init)
            dompc_state_data = np.empty([num_iterations,13])
            dompc_control_data = np.empty([num_iterations,4])
            dompc_time_data = []
            x0 = x_init

            # run do-mpc solver
            # print('running do-mpc solver')
            for k in range(num_iterations):
                start_time = perf_counter()
                mpc.set_waypoint(np.array([xr[0], xr[1], xr[2], mc.battery_v, mc.hover_thrust]))
                u0 = mpc.mpc.make_step(x0)
                step_time = perf_counter() - start_time

                # runge kutta 4 simulator
                x0 = rk_sim.make_step(ms_model.f, x0, u0, params)

                dompc_state_data[k] = np.reshape(x0, (13,))
                dompc_control_data[k] = np.reshape(u0, (4,))
                dompc_time_data.append(step_time)
                

            # compute the accuracy metric
            state_error = 0
            for i in range(num_iterations):
                error = reference_data[i] - dompc_state_data[i]
                state_error += error.T @ error

            accuracy[c-1][n] = sig_figs(state_error, 2)

            # compute statistics for the timing of the nmpc calls
            mean_time = sig_figs(stats.mean(dompc_time_data),2)
            max_time = round(max(dompc_time_data),3)
            times[c-1][n] = mean_time
            from experiments.trajectory_metrics import settling_metric
            settle = round(settling_metric(dompc_state_data, goal_ll, goal_ul) * mc.dt, 3) 
            s = ["{: >20} ".format(v) for v in [tstep, c, mean_time, state_error, settle]]
            print(''.join(s))



            # uncomment if you want to see plots of the trajectories
            plot_state_for_paper(tspan, dompc_state_data, test["title"], 1)
            plot_control_for_paper(tspan, dompc_control_data, test["title"], 2)
            plt.show()


    # here print out our results to std out
    header = '    '
    for t in timesteps:
        header = header + str(t).rjust(8)
    print(header)
    print('------------------------------------------------------------')
    for c in range(len(collocations)):
        print(str(c) + ' | ', end='')
        for t in range(len(timesteps)):
            print(str(times[c][t]).rjust(8), end='')
        print()

    c_labels = [str(c) for c in collocations]
    t_labels = [str(t) for t in timesteps]

    print(accuracy)

    plotdir = "plots/"
    # create a heat plot 
    # plt.figure(1)
    fig, ax = plt.subplots()
    # fig.set_figheight(8)
    im = ax.imshow(times, norm=colors.LogNorm(vmin=times.min(), vmax=times.max()), cmap='viridis')

    ax.set_xticks(np.arange(len(t_labels)))
    ax.set_yticks(np.arange(len(c_labels)))
    ax.set_xticklabels(t_labels)
    ax.set_yticklabels(c_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(c_labels)):
        for j in range(len(t_labels)):
            text = ax.text(j, i, str(times[i, j]),
                        ha="center", va="center", color="w")

    ax.set_title("Average CPU time")

    plt.savefig(plotdir + "CPU.pdf", format="pdf", bbox_inches="tight")

    
    fig1, ax1 = plt.subplots()
    im2 = ax1.imshow(accuracy, norm=colors.LogNorm(), cmap='viridis')

    ax1.set_xticks(np.arange(len(t_labels)))
    ax1.set_yticks(np.arange(len(c_labels)))
    ax1.set_xticklabels(t_labels)
    ax1.set_yticklabels(c_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(c_labels)):
        for j in range(len(t_labels)):
            text = ax1.text(j, i, accuracy[i, j],
                        ha="center", va="center", color="w")

    ax1.set_title("Accuracy") 

    plt.savefig(plotdir + "Accuracy.pdf", format="pdf", bbox_inches="tight")

    
    plt.show()

    # plt.style.use('_mpl-gallery-nogrid')
    # plt.imshow(log_times, cmap='coolwarm', origin='lower') # 'origin' can be 'upper' or 'lower'
    # plt.colorbar(label='Value')
    # plt.title('Average CPU time')
    # plt.xlabel('time step')
    # plt.ylabel('collocation degree')
    # plt.show()


    