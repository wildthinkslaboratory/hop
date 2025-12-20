#
# This runs experiments on do-mpc's orthogonal collocation method to 
# determine the optimal number of intervals and the number of collocation points.
# This should produce two 2D heat maps one for accuracy and one for time
# with the 2D grid corresponding to combinations of number of collocation points
# and size of intervals
#
from hop.drone_model import DroneModel
from hop.drone_model_randomized import DroneModelRandom
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
from do_mpc.simulator import Simulator
import casadi as ca
from do_mpc.estimator import StateFeedback
import numpy as np
import statistics as stats
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib import colors
from plots import plot_comparison, plot_state_for_paper, plot_control_for_paper
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
    "title": "x1z1vx"
  }
]


test = test_list[0]
terminal_error = []

for i in range(10):
    
    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # run fine grained solver for a reference trajectory
    # the accuracy of other runs are assessed relative to this trajectory
    model = DroneModel()
    modelRandom = DroneModelRandom()
    mpc = DroneNMPCdompc(mc.dt, model.model)

    mpc.mpc.settings.t_step = 0.3
    mpc.mpc.settings.n_horizon = int(2.0 / 0.3)
    mpc.mpc.settings.collocation_deg = 2

    estimator = StateFeedback(model.model)
    sim = Simulator(modelRandom.model)
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
    mpc.setup_cost()
    mpc.set_start_state(x_init)
    sim.x0 = x_init
    estimator.x0 = x_init
    reference_data = np.empty([num_iterations,13])
    x0 = x_init
    time_data = []

    # print('running reference do-mpc solver')
    for k in range(num_iterations):
        start_time = perf_counter()
        u0 = mpc.mpc.make_step(x0)
        step_time = perf_counter() - start_time

        y_next = sim.make_step(u0)
        x0 = estimator.make_step(y_next)
        reference_data[k] = np.reshape(x0, (13,))
        time_data.append(step_time)
    
    error = xr - x0
    squared_error = error.T @ error
    terminal_error.append(squared_error)

    print('terminal error', squared_error)
    # uncomment if you want to see plots of the trajectories
    print(test["title"])
    plot_state_for_paper(tspan, reference_data, test["title"], 1)
    # plot_control_for_paper(tspan, reference_data, test["title"], 2)
    plt.show()

error = xr - x_init
print(error.T @ error)
print(terminal_error)

