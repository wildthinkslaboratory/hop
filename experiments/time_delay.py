# This code experiments with time delays. 
# Simulate having a time delay in the control
# Then look at how different NLP models do with the time delay
#

from hop.constants import Constants
from hop.equations_of_motion import Equations6DOF
from hop.utilities import  import_data
import casadi as ca
import numpy as np
from simulation_tools.integrators import RKSimulator
from time import perf_counter
from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc
import matplotlib.pyplot as plt
from plotting.plots import plot_state_for_paper, plot_control_for_paper, plot_comparison


from hop.multiShooting import DroneNMPCMultiShoot
from hop.multiShootTDelay import DroneNMPCMultiShootTDelay




# first we make a model
mc = Constants()
equations = Equations6DOF(mc)



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

delay_steps = 3


for test in test_list_for_paper:

    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = np.array(test['xr'])
    allowed_error = np.array([0.05,0.05,0.05, 0.02,0.02,0.02, 0.02,0.02,0.02,0.02, 0.01,0.01,0.01])
    goal_ul = xr + allowed_error
    goal_ll = xr - allowed_error

    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)
    params = np.array([xr[0], xr[1], xr[2], mc.battery_v, mc.hover_thrust])

    state_data = np.empty([num_iterations,13])
    control_data = np.empty([num_iterations,4])
    time_data = []
    # cost_data = []
    stats_data = 0
    
    # set up the Runge-Kutta simulator
    rk_sim = RKSimulator(0.005, 4)
    params = np.array([xr[0], xr[1], xr[2], mc.battery_v, mc.hover_thrust])
    
    model = DroneModel(mc)
    mpc = DroneNMPCdompc(mc.dt, model.model)
    mpc.setup_cost()
    mpc.set_start_state(x_init)

    # model_delay = True

    # if model_delay:
    #     ms_nmpc = DroneNMPCMultiShootTDelay(equations, delay_steps)
    #     ms_nmpc.build_nmpc_instance()
    #     ms_nmpc.set_start_state(x_init)
    #     u0 = np.zeros(4)
    # else:
    #     ms_nmpc = DroneNMPCMultiShoot(equations)
    #     ms_nmpc.build_nmpc_instance()
    #     ms_nmpc.set_start_state(x_init)
    #     u0 = np.zeros(4)



    x0 = x_init


    u_history = np.tile([0.0, 0.0, mc.hover_thrust, 0.0], (delay_steps, 1))
    print(u_history)

    # run the simulation
    for k in range(num_iterations):
        start_time = perf_counter()

        mpc.set_waypoint(params)
        u0 = mpc.mpc.make_step(x0)

        
        # if model_delay:
        #     u0 = ms_nmpc.make_step(x0, u_history.flatten(), params)
        # else:
        #     u0 = ms_nmpc.make_step(x0, u0, params)


        step_time = perf_counter() - start_time

        active_u = u0

        # keep a history of control vectors
        if delay_steps > 0:
            active_u = u_history[delay_steps-1]
            u_history = np.roll(u_history, 1, axis=0)
            u_history[0] = u0.flatten()



        # runge kutta 4 simulator
        x0 = rk_sim.make_step(equations.f, x0, active_u, params)

        state_data[k] = np.reshape(x0, (13,))
        control_data[k] = np.reshape(u0, (4,))
        time_data.append(step_time)

        if not mpc.mpc.solver_stats['return_status'] == 'Solve_Succeeded':
            stats_data += 1

        # if not ms_nmpc.solver_stats['status'] == 'Solve_Succeeded':
        #     stats_data += 1
        #     print("NPL fail on iteration: ", k)


    # state_error = 0
    # for i in range(num_iterations):
    #     error = reference_data[i] - state_data[i]
    #     state_error += error.T @ error


    # accuracy = sig_figs(state_error, 2)
    # from experiments.trajectory_metrics import settling_metric
    # settle = round(settling_metric(state_data, goal_ll, goal_ul) * mc.dt, 3) 
    # mean_time = round(stats.mean(time_data),3)
    # s = ["{: >20} ".format(v) for v in [ms_nmpc.dt, ms_nmpc.N, mean_time, accuracy, settle]]
    # print(''.join(s))

    print("NLP fails: ", stats_data)

    plot_state_for_paper(tspan, state_data, test["title"], 'ms', 1)
    plot_control_for_paper(tspan, control_data, test["title"], 'ms', 2)
    
    times = {'ms': time_data}
    plot_comparison(tspan, times, test["title"], 3, 'CPU Time (sec)')

    plt.show()

#################################################################################################


 
