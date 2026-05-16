# Experiementing with which drone constants affect stability and performance
#
from hop.drone_model import DroneModel
from hop.multiShooting import DroneNMPCMultiShoot
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
from plots import plot_comparison, plot_state_for_sensitivity, plot_control_for_sensitivity
from hop.utilities import sig_figs
mc = Constants()


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


    Ixx =  0.0586     # these are all +- 0.0005
    Iyy =  0.0590     
    Izz =  0.0126     
    Ixz =  0.0003
    Iyz =  0.0010

    x_moment = mc.moment_arm[0] #  0.000045 -> -0.000147, 0.001284, 0.000422
    y_moment = mc.moment_arm[1] #  0.000033 -> -0.000346, 0.000453, 0.000006
    z_moment = mc.moment_arm[2] #  0.211626 ->  0.211713, 0.212190, 0.211493

    mar = 0.002
    mir = 0.002

    for i in range(5):

        mc.moment_arm[0] = x_moment 
        mc.moment_arm[1] = y_moment 
        mc.moment_arm[2] = z_moment 
        ms_believed = DroneNMPCMultiShoot(mc)


        mc.moment_arm[0] = x_moment + np.random.uniform(-mar, mar)
        mc.moment_arm[1] = y_moment + np.random.uniform(-mar, mar)
        mc.moment_arm[2] = z_moment + np.random.uniform(-mar, mar)

        print('moment arm', mc.moment_arm)

        # Ixx =  0.0586 + np.random.uniform(-mir, mir)
        # Iyy =  0.0590 + np.random.uniform(-mir, mir)   
        # Izz =  0.0126 + np.random.uniform(-mir, mir)
        # Ixz =  0.0003 + np.random.uniform(-mir, mir)
        # Iyz =  0.0010 + np.random.uniform(-mir, mir)

        # mc.I = np.array([
        #     [Ixx, 0.0, Ixz],
        #     [0.0, Iyy, Iyz],
        #     [Ixz, Iyz, Izz]
        # ])

        # print('I', mc.I.flatten().tolist())


        ms_actual = DroneNMPCMultiShoot(mc)
        ms_actual.record_nlp_stats = True
        ms_actual.set_goal_state(xr)
        ms_actual.set_start_state(x_init)

        state_data = np.empty([num_iterations,13])
        control_data = np.empty([num_iterations, 4])
        x0 = x_init
        u0 = np.zeros(4)
        time_data = []
        cost_data = []

        # print('running reference do-mpc solver')
        for k in range(num_iterations):
            start_time = perf_counter()
            u0 = ms_actual.make_step(x0, u0, np.array([0.0, 0.0, 0.0]))
            step_time = perf_counter() - start_time

            # Propagate the system using the discrete dynamics f (Euler forward integration)
            x0 = x0 + mc.dt* ms_believed.f(x0,u0)

            state_data[k] = np.reshape(x0, (13,))
            control_data[k] = np.reshape(u0, (4,))
            time_data.append(step_time)
            cost_data.append(ms_actual.solver_stats['cost'])
            if not ms_actual.solver_stats['status'] == 'Solve_Succeeded':
                print(ms_actual.solver_stats['status'])


        # # uncomment if you want to see plots of the trajectories
        print(test["title"])
        plot_state_for_sensitivity(tspan, state_data, test["title"], 1)
        plot_control_for_sensitivity(tspan, control_data, test["title"], 2)
        plt.show()

