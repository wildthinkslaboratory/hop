from flight_analysis_tools.flight_data import FlightData
from hop.constants import Constants
from simulation_tools.integrators import RKSimulator
from hop.equations_of_motion import Equations6DOF
from plotting.plots import plot_state, trajectory_comparison
from hop.utilities import quaternion_to_angle
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


fd = FlightData()
mc = Constants()
mc.update_from_dictionary(fd.constants)

show_horizon_trajectory = False


##########################################################
# If you want to mess with any constants to see if you 
# can get a better fit to the flight data, do it here



##########################################################
delay = mc.nmpc_delay

horizon_steps = 25 # int(mc.horizon_time / mc.dt)
equations = Equations6DOF(mc)

rk_sim1 = RKSimulator(0.005, 4)


residual_1 = np.zeros([fd.len_used_data-1,13])
residual_delay = np.zeros([fd.len_used_data-1,13])
residual_horizon = np.zeros([fd.len_used_data-1,13])

predicted_dx = np.zeros([fd.len_used_data-1,13])
full_predicted_dx = np.zeros([fd.len_used_data-1,13])
actual_dx = np.zeros([fd.len_used_data-1,13])
roll_dx = np.zeros([fd.len_used_data-1,13])
predicted_angle = np.zeros([fd.len_used_data-1,3])
actual_angle = fd.attitude[:-2]
zero = np.zeros([fd.len_used_data-1,1])

model_dvz = []
actual_dvz = []

roll = 5
x_history = deque(maxlen=roll)

for i in range(delay, len(fd.state_data)-1):

    # x = fd.state_data[i][0]  
    # y = fd.state_data[i][1]
    # z = fd.state_data[i][2]
    # r_xy = np.sqrt(x**2 + y**2)
    


    #####################################################################
    predicted_state = np.reshape(rk_sim1.make_step(equations.f, fd.state_data[i], fd.control_data[i-delay], fd.parameters[i]), (13,))
    residual_1[i] = predicted_state - fd.state_data[i+1]


    if (i < len(fd.state_data) - delay):
        state = fd.state_data[i]
        for j in range(delay):
            state = rk_sim1.make_step(equations.f, state, fd.control_data[i-delay+j], fd.parameters[i+j])
        residual_delay[i] = np.reshape(state, (13,)) - fd.state_data[i+delay]


    if (i < len(fd.state_data) - horizon_steps):
        horizon_traj = np.zeros([horizon_steps+1, 13])
        state = fd.state_data[i]
        horizon_traj[0] = fd.state_data[i]
        
        for j in range(horizon_steps):
            state = rk_sim1.make_step(equations.f, state, fd.control_data[i-delay+j], fd.parameters[i+j])
            horizon_traj[j+1] = np.reshape(state, (13,))
        residual_horizon[i] = np.reshape(state, (13,)) - fd.state_data[i+horizon_steps]

        if show_horizon_trajectory:
            tspan = np.arange(0, (horizon_steps+1) * mc.dt, mc.dt)
            trajectory_comparison(tspan, horizon_traj, tspan, fd.state_data[i:i+horizon_steps+1])
            plot_state(tspan, fd.state_data[i:i+horizon_steps+1], 'actual flight data')
            plot_state(tspan, horizon_traj, 'predicted trajectory')
            plt.show()


    full_predicted_dx[i] = np.reshape(equations.f(fd.future_state_data[i-delay], fd.control_data[i-delay], fd.parameters[i-delay]) * mc.dt, (13,))
    predicted_dx[i] = np.reshape(equations.f(fd.state_data[i], fd.control_data[i-delay], fd.parameters[i]) * mc.dt, (13,))
    actual_dx[i] = np.reshape(fd.state_data[i+1] - fd.state_data[i], (13,))
    x_history.append(actual_dx[i].copy())
    back = int(roll / 2)
    if i >= back: 
        roll_dx[i-back] = np.mean(x_history, axis=0)

    model_dvz.append(predicted_dx[i][5])
    actual_dvz.append(actual_dx[i][5])

    # turn quaternions into attitude
    q = np.reshape(predicted_state[6:10].copy(), (4,))
    predicted_angle[i] = quaternion_to_angle(q)

tspan = np.arange(0, (fd.len_used_data-1) * fd.dt , fd.dt)
if len(tspan) > len(predicted_dx):
    tspan = tspan[:-1]


fig, axs = plt.subplots(3)
fig.set_figheight(8)
fig.suptitle('angular velocity differential comparison')

# axs[0].plot(tspan, full_predicted_dx[:,10])
axs[0].plot(tspan, predicted_dx[:,10])
# axs[0].plot(tspan, actual_dx[:,10])
axs[0].plot(tspan, roll_dx[:,10])
axs[0].plot(tspan, zero)
axs[0].set_ylabel('$w_x$')

# axs[1].plot(tspan, full_predicted_dx[:,10])
axs[1].plot(tspan, predicted_dx[:,11])
# axs[1].plot(tspan, actual_dx[:,11])
axs[1].plot(tspan, roll_dx[:,11])
axs[1].plot(tspan, zero)
axs[1].set_ylabel('$w_y$')

# axs[2].plot(tspan, full_predicted_dx[:,10])
axs[2].plot(tspan, predicted_dx[:,12])
# axs[2].plot(tspan, actual_dx[:,12])
axs[2].plot(tspan, roll_dx[:,12])
axs[2].plot(tspan, zero)
axs[2].set_ylabel('$w_z$')
plt.xlabel('Time')

fig, axs = plt.subplots(3)
fig.set_figheight(8)
fig.suptitle('velocity differentials comparison')


axs[0].plot(tspan, actual_dx[:,3])
axs[0].plot(tspan, predicted_dx[:,3])
axs[0].plot(tspan, roll_dx[:,3])
axs[0].plot(tspan, zero)
axs[0].set_ylabel('$v_x$')

axs[1].plot(tspan, actual_dx[:,4])
axs[1].plot(tspan, predicted_dx[:,4])
axs[1].plot(tspan, roll_dx[:,4])
axs[1].plot(tspan, zero)
axs[1].set_ylabel('$v_y$')

axs[2].plot(tspan, actual_dx[:,5])
axs[2].plot(tspan, predicted_dx[:,5])
# axs[2].plot(tspan, roll_dx[:,5])
axs[2].plot(tspan, zero)
axs[2].set_ylabel('$v_z$')
plt.xlabel('Time')

# fig, axs = plt.subplots(3)
# fig.set_figheight(8)
# fig.suptitle('predicted vs actual attitude')

# axs[0].plot(tspan, predicted_angle[:,0])
# axs[0].plot(tspan, actual_angle[:,0])
# axs[0].set_ylabel('$x$')

# # y angle
# axs[1].plot(tspan, predicted_angle[:,1])
# axs[1].plot(tspan, actual_angle[:,1])
# axs[1].set_ylabel('$y$')

# # total angle
# axs[2].plot(tspan, predicted_angle[:,2])
# axs[2].plot(tspan, actual_angle[:,2])
# axs[2].set_ylabel('tilt')

# plt.xlabel('Time')


plot_state(tspan, residual_1, 'predicted state minus actual')
plot_state(tspan, residual_delay, 'delay steps state minus actual')
plot_state(tspan, residual_horizon, 'horizon steps state minus actual')

plt.figure(6)
plt.scatter(model_dvz, actual_dvz)

m, b = np.polyfit(model_dvz, actual_dvz, 1)

xfit = np.linspace(np.min(model_dvz),
                   np.max(model_dvz), 100)

plt.plot(xfit, m*xfit + b,
         label=f"fit: y={m:.2f}x+{b:.4f}")

plt.xlabel("Predicted Δv_z")
plt.ylabel("Actual Δv_z")

# Let matplotlib autoscale first
plt.autoscale()

# Now get the visible limits
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

# Draw y=x only over the overlapping visible range
lo = max(xmin, ymin)
hi = min(xmax, ymax)

plt.plot([lo, hi], [lo, hi], 'k--', label="y = x")

plt.legend()
plt.show()

plt.show()