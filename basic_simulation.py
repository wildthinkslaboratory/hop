#
#
from hop.drone_model import DroneModel
from hop.drone_model_randomized import DroneModelRandom
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
from do_mpc.simulator import Simulator
from utilities import import_data
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

print(mc.__dict__)

test_list = import_data('./nmpc_test_cases.json')  

# test_list = [
#   {
#     "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#     "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#     "animation_forward": [0.0, -0.2, -1],
#     "animation_up": [0, 1, 0],
#     "animation_frame_rate": 0.8,
#     "num_iterations": 200,
#     "title": "hover"
#   },
# ]


for test in test_list:

    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # run fine grained solver for a reference trajectory
    # the accuracy of other runs are assessed relative to this trajectory
    model = DroneModel(mc)
    mpc = DroneNMPCdompc(mc.dt, model.model)

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
    mpc.setup_cost()
    mpc.set_start_state(x_init)
    sim.x0 = x_init
    estimator.x0 = x_init
    state_data = np.empty([num_iterations,13])
    control_data = np.empty([num_iterations, 4])
    x0 = x_init
    time_data = []
    cost_data = []

    # print('running reference do-mpc solver')
    for k in range(num_iterations):
        start_time = perf_counter()
        # mpc.set_waypoint(np.array([0.0, 0.0, 0.0]))
        u0 = mpc.mpc.make_step(x0)
        step_time = perf_counter() - start_time

        y_next = sim.make_step(u0)
        x0 = estimator.make_step(y_next)
        state_data[k] = np.reshape(x0, (13,))
        control_data[k] = np.reshape(u0, (4,))

        time_data.append(step_time)
        cost_data.append(mpc.mpc.data['_aux'][-1][2])
        if not mpc.mpc.solver_stats['return_status'] == 'Solve_Succeeded':
            print(mpc.mpc.solver_stats['return_status'])


    
    # error = xr - x0
    # squared_error = error.T @ error
    # terminal_error.append(squared_error)

    # print('terminal error', squared_error)
    # # uncomment if you want to see plots of the trajectories
    print(test["title"])
    plot_state_for_paper(tspan, state_data, test["title"], 1)
    plot_control_for_paper(tspan, control_data, test["title"], 2)
    plt.show()

# error = xr - x_init
# print(error.T @ error)
# print(terminal_error)

