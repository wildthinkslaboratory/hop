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
from time import perf_counter
import matplotlib.pyplot as plt
from plots import plot_time_comparison, plot_state_for_paper, plot_control_for_paper
from math import log
mc = Constants()


# plt.style.use('_mpl-gallery-nogrid')
# # make data
# X, Y = np.meshgrid(np.linspace(-3, 3, 16), np.linspace(-3, 3, 16))
# Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
# plt.imshow(Z, cmap='coolwarm', origin='lower') # 'origin' can be 'upper' or 'lower'
# plt.colorbar(label='Value')
# plt.title('Custom Heatmap with imshow')
# plt.xlabel('collocation degree')
# plt.ylabel('time step')
# plt.show()



# If you just want to run a single test you can loop over this list
test_list = [
  {
    "x0": [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-1, -0.1, -0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 200,
    "title": "x=1 y=1 z=0.5 15deg rotation "
  }
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

times = np.zeros((5,7))
accuracy = times
collocations = [1, 2, 3, 4, 5]
timesteps = [0.02, 0.04, 0.25, 0.5, 1.0, 2.0]

for test in test_list:
    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # run fine grained solver for a reference trajectory
    model = DroneModel()
    mpc = DroneMPC(mc.dt, model.model)

    mpc.mpc.settings.t_step = 0.02
    mpc.mpc.settings.n_horizon = 100
    mpc.mpc.settings.collocation_deg = 2 

    estimator = StateFeedback(model.model)
    sim = Simulator(model.model)
    sim.set_param(t_step = mc.dt)
    sim.setup()
    mpc.set_goal_state(xr)
    mpc.set_start_state(x_init)
    sim.x0 = x_init
    estimator.x0 = x_init
    reference_data = np.empty([num_iterations,13])
    x0 = x_init

    # run do-mpc solver
    print('running reference do-mpc solver')
    for k in range(num_iterations):
        start_time = perf_counter()
        u0 = mpc.mpc.make_step(x0)
        step_time = perf_counter() - start_time

        y_next = sim.make_step(u0)
        x0 = estimator.make_step(y_next)
        reference_data[k] = np.reshape(x0, (13,))

                

    for c in collocations:
        for n,tstep in enumerate(timesteps):
            n_horizon = int(2.0 / tstep)

            # first we set up the do-mpc solver
            # it uses orthagonal collocation
            model = DroneModel()
            mpc = DroneMPC(mc.dt, model.model)

            mpc.mpc.settings.t_step = tstep
            mpc.mpc.settings.n_horizon = n_horizon
            mpc.mpc.settings.collocation_deg = c  

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
            # print('running do-mpc solver')
            for k in range(num_iterations):
                start_time = perf_counter()
                u0 = mpc.mpc.make_step(x0)
                step_time = perf_counter() - start_time

                y_next = sim.make_step(u0)
                x0 = estimator.make_step(y_next)

                dompc_state_data[k] = np.reshape(x0, (13,))
                dompc_control_data[k] = np.reshape(u0, (4,))
                dompc_time_data.append(step_time)
                


            # compute the accuracy metric
            state_error = 0
            for i in range(num_iterations):
                error = reference_data[i] - dompc_state_data[i]
                state_error += error.T @ error
            accuracy[c-1][n] = round(state_error, 3)

            # compute statistics for the timing of the nmpc calls
            mean_time = round(stats.mean(dompc_time_data),3)
            max_time = round(max(dompc_time_data),3)
            times[c-1][n] = mean_time
            print(c, tstep, state_error, mean_time)

            # print timing results
            print(test['title'])
            print('mean: ', mean_time, ' max: ', max_time)


            plot_state_for_paper(tspan, dompc_state_data, test["title"], 1)
            plot_control_for_paper(tspan, dompc_control_data, test["title"], 2)
            plt.show()

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

    c_labels = ['1', '2', '3', '4', '5']
    t_labels = ['0.02', '0.04', '0.08', '0.25', '0.5', '1.0', '2.0']



    fig, ax = plt.subplots(2)
    fig.set_figheight(12)
    from matplotlib.colors import LogNorm
    im = ax[0].imshow(times)

    ax[0].set_xticks(np.arange(len(t_labels)))
    ax[0].set_yticks(np.arange(len(c_labels)))
    ax[0].set_xticklabels(t_labels)
    ax[0].set_yticklabels(c_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(c_labels)):
        for j in range(len(t_labels)):
            text = ax[0].text(j, i, str(times[i, j]),
                        ha="center", va="center", color="w")

    ax[0].set_title("Average CPU time")

    im2 = ax[1].imshow(accuracy)

    ax[1].set_xticks(np.arange(len(t_labels)))
    ax[1].set_yticks(np.arange(len(c_labels)))
    ax[1].set_xticklabels(t_labels)
    ax[1].set_yticklabels(c_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(c_labels)):
    #     for j in range(len(t_labels)):
    #         text = ax[1].text(j, i, accuracy[i, j],
    #                     ha="center", va="center", color="w")

    ax[1].set_title("Accuracy") 

    # fig.tight_layout()

    
    plt.show()

    # plt.style.use('_mpl-gallery-nogrid')
    # plt.imshow(log_times, cmap='coolwarm', origin='lower') # 'origin' can be 'upper' or 'lower'
    # plt.colorbar(label='Value')
    # plt.title('Average CPU time')
    # plt.xlabel('time step')
    # plt.ylabel('collocation degree')
    # plt.show()


    