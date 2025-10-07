#
# This runs experiments on do-mpc's orthogonal collocation method to 
# determine the optimal number of intervals and the number of collocation points.
# This should produce two 2D heat maps one for accuracy and one for time
# with the 2D grid corresponding to combinations of number of collocation points
# and size of intervals
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
from matplotlib import colors
from plots import plot_time_comparison, plot_state_for_paper, plot_control_for_paper
from hop.utilities import sig_figs
mc = Constants()


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
]

collocations = [1, 2, 3, 4, 5]                        # number of collocation points
timesteps = [0.02, 0.04, 0.08, 0.2, 0.25, 0.5, 0.8]   # size of the time intervals (within a 2 second horizon)

times = np.zeros((len(collocations),len(timesteps)))     # we measure the average solution time
accuracy = np.zeros((len(collocations),len(timesteps)))  # we measure the accuracy

for test in test_list:
    
    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # run fine grained solver for a reference trajectory
    # the accuracy of other runs are assessed relative to this trajectory
    model = DroneModel()
    mpc = DroneMPC(mc.dt, model.model)

    mpc.mpc.settings.t_step = 0.02
    mpc.mpc.settings.n_horizon = 100
    mpc.mpc.settings.collocation_deg = 5 

    estimator = StateFeedback(model.model)
    sim = Simulator(model.model)
    sim.set_param(t_step = mc.dt)

    # this is annoying but necessary
    # the model has a parameter for waypoints
    # it's used in the mpc cost function only
    # the simulator get's confused if it has undefined parameters
    # so we give it a dummy function
    p_template = sim.get_p_template()
    def dummy(t_now):
        p_template['p_goal'] = np.array([0.0, 0.0, 0.0])
        return p_template
    sim.set_p_fun(dummy)

    sim.setup()
    mpc.set_goal_state()
    mpc.set_start_state(x_init)
    sim.x0 = x_init
    estimator.x0 = x_init
    reference_data = np.empty([num_iterations,13])
    x0 = x_init

    print('running reference do-mpc solver')
    for k in range(num_iterations):
        start_time = perf_counter()
        u0 = mpc.mpc.make_step(x0)
        step_time = perf_counter() - start_time

        y_next = sim.make_step(u0)
        x0 = estimator.make_step(y_next)
        reference_data[k] = np.reshape(x0, (13,))

                
    # These are our experimental runs 
    # loop over the number of collocation and 
    # size of the time intervals
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
            sim.set_p_fun(dummy)
            sim.setup()
            mpc.set_goal_state()
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

            accuracy[c-1][n] = sig_figs(state_error, 2)

            # compute statistics for the timing of the nmpc calls
            mean_time = sig_figs(stats.mean(dompc_time_data),2)
            max_time = round(max(dompc_time_data),3)
            times[c-1][n] = mean_time
            print(c, tstep, state_error, mean_time)

            
            # print timing results
            # print(test['title'])
            # print('mean: ', mean_time, ' max: ', max_time)

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

    # create a heat plot 
    # plt.figure(1)
    fig, ax = plt.subplots()
    fig.set_figheight(8)
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

    # plt.savefig("CPU.pdf", format="pdf", bbox_inches="tight")

    plt.figure(2)
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

    # plt.savefig("Accuracy.pdf", format="pdf", bbox_inches="tight")

    
    plt.show()

    # plt.style.use('_mpl-gallery-nogrid')
    # plt.imshow(log_times, cmap='coolwarm', origin='lower') # 'origin' can be 'upper' or 'lower'
    # plt.colorbar(label='Value')
    # plt.title('Average CPU time')
    # plt.xlabel('time step')
    # plt.ylabel('collocation degree')
    # plt.show()


    