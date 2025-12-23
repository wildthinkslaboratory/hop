# This is an analysis of our maiden voyage
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

test_list = [
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [0.0, -0.2, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.8,
    "num_iterations": 500,
    "title": "hover"
  },
]


for test in test_list:

    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # first we set up the constants used in our maiden voyage
    # and create a model with those constants
    mc.m = 1.5
    mc.moment_arm = np.array([0.0, 0.0, -0.18 / 2])
    mc.I = np.array([
        [0.0600, 0.0000, 0.0000],
        [0.0000, 0.0600, 0.0000],
        [0.0000, 0.0000, 0.0120]
        ])
    model_believed = DroneModel(mc)

    # now we create a model with constants that more closely
    # resemble the actual real world values
    mc.m = 1.601
    mc.moment_arm = np.array([0.005, -0.001, -0.21])
    mc.I = np.array([
        [0.0595, 0.0000, 0.0019],
        [0.0000, 0.0601, 0.0009],
        [0.0019, 0.0009, 0.0133]
        ])
    model_actual = DroneModel(mc)

    # here's were we can try different things
    #
    # 1. try using the model_believed for the mpc, estimator and simulator
    #    you will see the behavior we expected
    #
    # 2. try using the model_actual for mpc, estimator and simulator
    #    You will see that the drone can handle these values, but the gimbal
    #    angles steady state is not zero to adjust for the off center COM
    #
    # 3. try using model_believed for mpc and estimator, but use model_actual
    #    for the simulator. The drone can't handle this. It can't seem to move in the
    #    positive Y direction and gets stuck over at -1.5 meters. Notice that the
    #    steady state gimbal values are about the same as in experiment #2.
    mpc = DroneNMPCdompc(mc.dt, model_believed.model)

    estimator = StateFeedback(model_believed.model)
    sim = Simulator(model_believed.model)
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


    
    # # uncomment if you want to see plots of the trajectories
    print(test["title"])
    plot_state_for_paper(tspan, state_data, test["title"], 1)
    plot_control_for_paper(tspan, control_data, test["title"], 2)
    plt.show()



