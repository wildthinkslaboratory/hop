# For a given control time step we look at


#  - The cost of state and control elements across the horizon
#  - A sensitivity analysis for each state and control element
#
#  - the cost of final solution plot



from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
from flight_analysis_tools.flight_data import FlightData
from hop.utilities import quaternion_to_angle
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from plotting.plots import plot_control, plot_state, trajectory_comparison
from time import perf_counter



state_names = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'q_x', 'q_y','q_z','q_0','w_x','w_y','w_z',]
control_names = ['theta 1', 'theta 2', 'P avg', 'P diff']
mc = Constants()
fd = FlightData()
mc.update_from_dictionary(fd.constants)
delay = mc.nmpc_delay
print(mc.tuning_info())

model = DroneModel(mc)  
mpc = DroneNMPCdompc(mc.dt, model.model)
mpc.setup_cost()
x_init = ca.DM(fd.state_data[delay])
mpc.set_start_state(x_init)
x_r = mc.xr.full().flatten()
u_r = mc.ur.full().flatten()

control_computed_diff = np.zeros([fd.len_used_data,4])
time_data = np.zeros([fd.len_used_data,1])
status = np.zeros([fd.len_used_data,1])

horizon_timesteps = int(mc.horizon_time / mc.dt)



for i in range(delay, len(fd.state_data)-1):

    start_time = perf_counter()
    mpc.set_waypoint(np.array(fd.parameters[i]))
    u = mpc.mpc.make_step(fd.future_state_data[i])
    step_time = perf_counter() - start_time

    time_data[i] = step_time
    control_computed_diff[i] = fd.control_data[i] - np.reshape(u, (4,))

    if not mpc.mpc.solver_stats['return_status'] == 'Solve_Succeeded':
        status[i] = 1
    print('Solver Status:', status[i], '  Solver cpu time: ', time_data[i])

    x_r[0:2] = fd.parameters[i][0:2]
    state_sol = mpc.mpc.data.prediction(('_x',))
    control_sol = mpc.mpc.data.prediction(('_u',))
    horizon = np.empty([len(state_sol[0]),13])
    state_cost = np.empty([13, len(state_sol[0])])
    u_horizon = np.empty([len(control_sol[0]),4])
    for k, state in enumerate(state_sol):
        for j, val in enumerate(state):
            horizon[j][k] = val
            state_cost[k][j] = (val - x_r[k]) * mc.Q[k,k] * (val - x_r[k])

    # collect attitude data across horizon
    attitude = np.empty([len(state_sol[0]),3])
    for k, state in enumerate(horizon):
        q = np.reshape(state[6:10].copy(), (4,))
        attitude[k] = quaternion_to_angle(q)

    # set the goal thrust
    u_r[2] = fd.parameters[i][4] *  mc.battery_v / fd.parameters[i][3] 
    control_cost = np.empty([4, len(control_sol[0])])
    for k, u in enumerate(control_sol):
        for j, val in enumerate(u):
            u_horizon[j][k] = val
            control_cost[k][j] = (val - u_r[k]) * mc.R[k,k] * (val - u_r[k])

    fis = mc.finite_interval_size
    horizon_length = (len(state_sol[0])-1) * fis
    hspan = np.arange(0, horizon_length + fis, fis)

    control_length = (len(control_sol[0])-1) * fis
    uspan = np.arange(0, control_length + fis, fis)

    end = min(fd.len_used_data, i + horizon_timesteps)
    state_data = fd.state_data[i:end]
    tspan = np.arange(0, len(state_data) * mc.dt, mc.dt)
    if not len(tspan) == len(state_data):
        tspan = tspan[:-1]

    trajectory_comparison(hspan, horizon, tspan, state_data)

    attitude_data = fd.attitude[i:end]

    plt.figure(5)
    plt.plot(hspan, horizon[:,0])
    plt.plot(tspan, attitude_data[:,0])
    plt.title('x tilt')


    plt.figure(6)
    plt.plot(hspan, horizon[:,1])
    plt.plot(tspan, attitude_data[:,1])
    plt.title('x tilt')



    plt.figure(figsize=(10, 6))

    plt.stackplot(
        hspan,
        state_cost,
        labels=state_names
    )

    plt.xlabel('Time in horizon (s)')
    plt.ylabel('State cost')
    plt.legend(loc='upper right')

    plt.figure(figsize=(10, 6))

    plt.stackplot(
        uspan,
        control_cost,
        labels=control_names
    )

    plt.xlabel('Time in horizon (s)')
    plt.ylabel('Control cost')
    plt.legend(loc='upper right')


    # time_steps = int(horizon_length / 0.02)
    # tspan = np.arange(0, horizon_length + mc.dt, mc.dt)

    # if time_steps > len(fd.state_data):
    #     tspan = np.arange(0, len(fd.state_data) * mc.dt, mc.dt)
    #     plot_state(tspan, fd.state_data, 'flight state data')
    #     plot_control(tspan, fd.control_data, 'flight control data')
    # else:
    #     plot_state(tspan, fd.state_data[i:i+time_steps], 'state')

    plot_state(hspan, horizon, 'state nmpc horizon')
    plot_control(uspan, u_horizon, 'control nmpc horizon')
    fd.plot(plots=['state'])
    plt.show()
tspan = np.arange(0, fd.len_used_data * fd.dt , fd.dt)

plt.figure(1)
plt.plot(tspan, status, label='Solver Status 0/1 Success/Fail', marker='+', linestyle='None', markersize=4)
plt.title('cpu time')

plt.figure(2)
plt.plot(tspan, time_data, label='CPU Time', marker='+', linestyle='None', markersize=4)
plt.title('cpu time')

plot_control(tspan, control_computed_diff, 'control computed difference')

plt.show()










