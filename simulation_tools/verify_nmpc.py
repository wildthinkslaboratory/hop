
from hop.drone_model import DroneModel
from hop.dompc import DroneNMPCdompc
from hop.constants import Constants
from simulation_tools.integrators import RKSimulator
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from plotting.plots import plot_state, plot_control
from hop.equations_of_motion import Equations6DOF

mc = Constants()
equations = Equations6DOF(mc)
rk_sim = RKSimulator(0.005, 50)

model = DroneModel(mc) 
mpc = DroneNMPCdompc(mc.dt, model.model)
mpc.setup_cost()


x0 = ca.DM([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
xr = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

params = np.array([0.0, 0.0, 0.0, mc.battery_v, mc.hover_thrust])
mpc.set_start_state(x0)

# simulate a few steps
for i in range(10):
    mpc.set_waypoint(params)
    u0 = mpc.mpc.make_step(x0)
    x0 = rk_sim.make_step(equations.f, x0, u0, params)


state_sol = mpc.mpc.data.prediction(('_x',))
control_sol = mpc.mpc.data.prediction(('_u',))
horizon = np.empty([len(state_sol[0]),13])
u_horizon = np.empty([len(state_sol[0]),4])
for k, state in enumerate(state_sol):
    for j, val in enumerate(state):
        horizon[j][k] = val

for k, u in enumerate(control_sol):
    for j, val in enumerate(u):
        u_horizon[j][k] = val



fi_size = mc.finite_interval_size
hspan = np.arange(0, len(state_sol[0]) * fi_size, fi_size)
plot_state(hspan, horizon, 'prediction horizon state')
plot_control(hspan, u_horizon, 'prediction horizon control')


# nmpc horizon is 6 * 0.25 = 1.5 sec
# now we simulate the control and see if we get the same state trajectory
tspan = np.arange(0, 6 * 0.25 , 0.25)
rk_horizon = np.empty([len(state_sol[0]),13])
rk_horizon[0] = horizon[0]
for i in range(6):
    x0 = rk_sim.make_step(equations.f, x0, u_horizon[i], params)
    rk_horizon[i+1] = np.reshape(x0, (13,)) 

plot_state(hspan, rk_horizon, 'rk simulation with horizon control')

plt.show()




