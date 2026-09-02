from hop.constants import Constants
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from flight_analysis_tools.flight_data import FlightData
from collections import deque
from simulation_tools.integrators import RKSimulator
from hop.equations_of_motion import Equations6DOF


def get_cost(pred, actual):
    cost = 0.0
    for i in range(3,6):
        cost += (pred[i] - actual[i])**2
    for i in range(10,13):
        cost += (pred[i] - actual[i])**2
    return cost




mc = Constants()
fd = FlightData()
mc.update_from_dictionary(fd.constants)
cost_surface = np.zeros([101,101])

offset1_vals = np.arange(0.0, 4.01, 0.1)
offset2_vals = np.arange(0.0, 4.01, 0.1)
cost = np.zeros((len(offset1_vals), len(offset2_vals)))


for j, valj in enumerate(offset1_vals):
    for k, valk in enumerate(offset2_vals):

        mc.gimbal_offset = [valj, valk]
        print(mc.gimbal_offset)
        equations = Equations6DOF(mc)
        rk_sim1 = RKSimulator(0.005, 4)
        delay = mc.nmpc_delay
        roll = 5
        x_history = deque(maxlen=roll)
        back = int(roll / 2)
        prev_thrust = mc.hover_thrust

        for i in range(len(fd.state_data) - back):
            x_history.append(fd.state_data[i].copy())

            if i > delay:
                u = fd.control_data[i-delay - back]
                x = fd.state_data[i - back]
                x[13] = prev_thrust
                p = fd.parameters[i - back]
                rolled_x = np.mean(x_history, axis=0)
                predicted_x = rk_sim1.make_step(equations.f, x, u, p)
                prev_thrust = predicted_x[13]
                cost[j][k] += get_cost(predicted_x, rolled_x)


print('done')
# Find minimum
i_min, j_min = np.unravel_index(np.argmin(cost), cost.shape)

best_offset1 = offset1_vals[i_min]
best_offset2 = offset2_vals[j_min]

print("Best offset 1:", best_offset1)
print("Best offset 2:", best_offset2)
print("Minimum cost:", cost[i_min, j_min])


# Make coordinate grid
O1, O2 = np.meshgrid(
    offset1_vals,
    offset2_vals,
    indexing='ij'
)

plt.figure(figsize=(8, 6))

contour = plt.contourf(
    O1,
    O2,
    cost,
    levels=50
)

plt.colorbar(contour, label="Cost")

plt.scatter(
    best_offset1,
    best_offset2,
    marker='x',
    s=100
)

plt.xlabel("Gimbal 1 offset (deg)")
plt.ylabel("Gimbal 2 offset (deg)")
plt.title("Gimbal Offset Cost Surface")

plt.show()




