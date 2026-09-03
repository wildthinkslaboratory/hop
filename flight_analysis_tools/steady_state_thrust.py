# this code builds up a model for the thrust once steady state has been reached.
# we restrict data points to those that aren't influenced by the tether and
# have a window of 10 timesteps where the average pwm has been held constant.

from hop.constants import Constants
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from flight_analysis_tools.flight_data import FlightData
from hop.utilities import quaternion_to_angle
from scipy.optimize import least_squares



def get_az(vz, dt):
    t = np.arange(5) * dt
    slope, _ = np.polyfit(t, vz, 1)
    return slope


# put all the flight logs that we want to analyze in this directory
directory = Path("steadystate")

all_thrust = []
ss_window = 10
max_p_range = 0.02
max_p_slope = 0.05 

thrust = []
voltage = []
p_top = []
p_bottom = []
p_avg = []
acceleration_z = []
old_thrust = []
delta_T = []
time = []
p_range = []
p_slope = []
flight_id = [] # we need to keep track of breaks in the data so our windows are contiguous flight
id = 0
last_id = 0


for file in directory.iterdir():
    mc = Constants()
    fd = FlightData(file)
    mc.update_from_dictionary(fd.constants)
    id += 1 # starting a new contiguous set of data points

    for i in range(len(fd.state_data) - 2):

        if i > 5:
            # estimate the thrust in Newtons
            v = fd.parameters[i][3]  # read the filtered voltage value from the parameters
            a_z_raw = get_az(fd.state_data[i-2:i+3, 5], mc.dt)

            # we account for any rotation of the drone
            x_theta, y_theta, theta = quaternion_to_angle(fd.state_data[i][6:10])
            T =  mc.m * (-mc.gz + a_z_raw) / np.cos(theta * np.pi / 180.0)
            all_thrust.append(T)

            # since the drone is flown on a tether, we restrict data
            # points to a circle around (x,y) point (0,0)
            # if we go too far from (0,0) the tether can pull on the drone
            # giving us bad readings
            # We exclude points with low z values for same reason
            x = fd.state_data[i][0]  
            y = fd.state_data[i][1]
            z = fd.state_data[i][2]
            r_xy = np.sqrt(x**2 + y**2)

            # we limit our data points to those that aren't being pulled by the tether.
            # so stay close to (x,y) = (0,0) and points above the tether height
            if i > (ss_window + mc.nmpc_delay) and abs(fd.control_data[i][3]) <= 0.08 and r_xy < 0.1 and z > 0.7:

                # to build our steady state points T_ss, we need points where the pwm hasn't changed
                # much in the past few steps
                p_avg_ss = fd.control_data[i-mc.nmpc_delay - ss_window: i-mc.nmpc_delay, 2]
                p_avg_range = np.max(p_avg_ss) - np.min(p_avg_ss)
                tspan = np.arange(10) * mc.dt
                p_avg_slope = np.polyfit(tspan, p_avg_ss, 1)[0]

                if p_avg_range < max_p_range and abs(p_avg_slope) < max_p_slope:
                    if not last_id + 1 == i: # if this point is not contiguous with last, it's a new flight
                        id += 1
                
                    delta_T.append(T - all_thrust[-2])
                    old_thrust.append(all_thrust[-2])
                    p_top.append(fd.pwm_motors[i-mc.nmpc_delay][0])
                    p_bottom.append(fd.pwm_motors[i-mc.nmpc_delay][1])
                    p_avg.append(fd.control_data[i-mc.nmpc_delay][2])
                    thrust.append(T)
                    voltage.append(v)
                    acceleration_z.append(a_z_raw)
                    time.append(i)
                    p_range.append(p_avg_range)
                    p_slope.append(p_avg_slope)
                    flight_id.append(id)
                    last_id = i


p_top = np.array(p_top)
p_bottom = np.array(p_bottom)
thrust = np.array(thrust)
voltage = np.array(voltage)
delta_T = np.array(delta_T)
p_avg = np.array(p_avg)
slope = np.array(p_slope)
p_range = np.array(p_range)

p_diff = (p_top - p_bottom) / 2
p_diff_abs = abs((p_top - p_bottom) / 2)
p_avg_scaled = p_avg * voltage / 25.0


import textwrap
##########################################################################
fig = plt.figure()
# look at relationship between average PWM and voltage
plt.scatter(thrust, p_avg, c=time, cmap='turbo', s=8)
plt.xlabel("thrust (N)")
plt.ylabel("P avg")
plt.colorbar(label="Voltage (V)")
note_text = (
    "Restricting to steady state points we see some structure from voltage and p_avg but the relationship isn't obvious."
)

note_text = textwrap.fill(note_text, width=70)

# Reserve room for the note
fig.subplots_adjust(bottom=0.20)

fig.text(
    0.1, 0.06,
    note_text,
    ha='left',
    va='top',
    fontsize=10
)
plt.show()


##########################################################################
A = np.column_stack((p_avg_scaled**2, p_avg_scaled, np.ones_like(p_avg_scaled)))
coeffs, _, _, _ = np.linalg.lstsq(A, thrust, rcond=None)

a, b, c = coeffs

print('a: ', a / 9.81)
print('b: ', b / 9.81)
print('c: ', c / 9.81)

predicted_thrust = A @ coeffs
error = thrust - predicted_thrust

rmse = np.sqrt(np.mean((error)**2))
r2 = 1 - np.sum((error)**2) / \
        np.sum((thrust - np.mean(thrust))**2)

print("coefficients:")
for i, c in enumerate(coeffs):
    print(f"c{i} = {c:.8f}")

print("RMSE:", rmse)
print("R²:", r2)
print("thrust std:", np.std(thrust))


#########################################################################


fig = plt.figure()

# plot predicted thrust vs. flight data thrust
plt.scatter(thrust, predicted_thrust, c=voltage, cmap='turbo', s=5)
plt.colorbar(label="voltage (V)")

lo = min(thrust.min(), predicted_thrust.min())
hi = max(thrust.max(), predicted_thrust.max())

plt.plot([lo, hi], [lo, hi], 'k--')

plt.xlabel("Measured thrust (N)")
plt.ylabel("Predicted thrust (N)")
plt.axis("equal")

note_text = (
    "Simple quadratic fit from p_avg scaled by voltage to thrust doesn't fit the data."
)

note_text = textwrap.fill(note_text, width=70)

# Reserve room for the note
fig.subplots_adjust(bottom=0.20)

fig.text(
    0.1, 0.06,
    note_text,
    ha='left',
    va='top',
    fontsize=10
)

plt.show()


#########################################################################

# plt.scatter(voltage, error,  s=8)
# plt.axhline(0, color='k', linestyle='--')
# plt.xlabel("voltage (V)")
# plt.ylabel("Thrust residual (N)")
# plt.show()




# ##########################################################################

# # we fit the data as a quadratic with pwm top, pwm bottom and voltage
# # all being independent of each other

X = np.column_stack([
    np.ones_like(p_avg),

    p_avg,
    voltage,

    p_avg**2,
    voltage**2,

    p_avg * voltage,
])

coeffs, *_ = np.linalg.lstsq(X, thrust, rcond=None)


predicted_thrust = X @ coeffs
error = thrust - predicted_thrust

rmse = np.sqrt(np.mean((error)**2))
r2 = 1 - np.sum((error)**2) / \
        np.sum((thrust - np.mean(thrust))**2)

print("coefficients:")
for i, c in enumerate(coeffs):
    print("self.c" + str(i) + " = " + str(c))

print("RMSE:", rmse)
print("R²:", r2)
print("thrust std:", np.std(thrust))


# ##########################################################################

fig = plt.figure()
# plot predicted thrust vs. flight data thrust
plt.scatter(thrust, predicted_thrust, c=time, cmap='turbo', s=5)
plt.colorbar(label="Voltage (V)")

lo = min(thrust.min(), predicted_thrust.min())
hi = max(thrust.max(), predicted_thrust.max())

plt.plot([lo, hi], [lo, hi], 'k--')

plt.xlabel("Measured thrust (N)")
plt.ylabel("Predicted thrust (N)")
plt.axis("equal")

note_text = (
    "Here we fit voltage and p_avg independently with 6 coefficients and we get more of the relationship"
)

note_text = textwrap.fill(note_text, width=70)

# Reserve room for the note
fig.subplots_adjust(bottom=0.20)

fig.text(
    0.1, 0.06,
    note_text,
    ha='left',
    va='top',
    fontsize=10
)

plt.show()

# ##########################################################################

# # we fit the data as a quadratic with pwm top, pwm bottom and voltage
# # all being independent of each other

X = np.column_stack([
    np.ones_like(p_top),

    p_avg,
    voltage,

    p_avg**2,
    voltage*2,
])

coeffs_5, *_ = np.linalg.lstsq(X, thrust, rcond=None)


predicted_thrust = X @ coeffs_5
error = thrust - predicted_thrust

rmse = np.sqrt(np.mean((error)**2))
r2 = 1 - np.sum((error)**2) / \
        np.sum((thrust - np.mean(thrust))**2)

print("coefficients:")
for i, c in enumerate(coeffs_5):
    print(f"c{i} = {c:.8f}")

print("RMSE:", rmse)
print("R²:", r2)
print("thrust std:", np.std(thrust))


# ##########################################################################

fig = plt.figure()
# plot predicted thrust vs. flight data thrust
plt.scatter(thrust, predicted_thrust, c=voltage, cmap='turbo', s=5)
plt.colorbar(label="Voltage (V)")

lo = min(thrust.min(), predicted_thrust.min())
hi = max(thrust.max(), predicted_thrust.max())

plt.plot([lo, hi], [lo, hi], 'k--')

plt.xlabel("Measured thrust (N)")
plt.ylabel("Predicted thrust (N)")
plt.axis("equal")

note_text = (
    "Here we fit voltage and p_avg independently with 5 coefficients but its quite a bit worse than the 6 coefficients"
)

note_text = textwrap.fill(note_text, width=70)

# Reserve room for the note
fig.subplots_adjust(bottom=0.20)

fig.text(
    0.1, 0.06,
    note_text,
    ha='left',
    va='top',
    fontsize=10
)

plt.show()

##########################################################################

thrust = []
voltage = []
p_top = []
p_bottom = []
p_avg = []
acceleration_z = []
old_thrust = []
delta_T = []
time = []
p_range = []
p_slope = []
flight_id = [] # we need to keep track of breaks in the data so our windows are contiguous flight
id = 0
last_id = 0


for file in directory.iterdir():
    mc = Constants()
    fd = FlightData(file)
    mc.update_from_dictionary(fd.constants)
    id += 1 # starting a new contiguous set of data points

    for i in range(len(fd.state_data) - 2):

        if i > 5:
            # estimate the thrust in Newtons
            v = fd.parameters[i][3]  # read the filtered voltage value from the parameters
            a_z_raw = get_az(fd.state_data[i-2:i+3, 5], mc.dt)

            # we account for any rotation of the drone
            x_theta, y_theta, theta = quaternion_to_angle(fd.state_data[i][6:10])
            T =  mc.m * (-mc.gz + a_z_raw) / np.cos(theta * np.pi / 180.0)
            all_thrust.append(T)

            # since the drone is flown on a tether, we restrict data
            # points to a circle around (x,y) point (0,0)
            # if we go too far from (0,0) the tether can pull on the drone
            # giving us bad readings
            # We exclude points with low z values for same reason
            x = fd.state_data[i][0]  
            y = fd.state_data[i][1]
            z = fd.state_data[i][2]
            r_xy = np.sqrt(x**2 + y**2)

            # we limit our data points to those that aren't being pulled by the tether.
            # so stay close to (x,y) = (0,0) and points above the tether height
            if i > (ss_window + mc.nmpc_delay) and abs(fd.control_data[i][3]) <= 0.08 and r_xy < 0.1 and z > 0.7:

                # to build our steady state points T_ss, we need points where the pwm hasn't changed
                # much in the past few steps
                p_avg_ss = fd.control_data[i-mc.nmpc_delay - ss_window: i-mc.nmpc_delay, 2]
                p_avg_range = np.max(p_avg_ss) - np.min(p_avg_ss)
                tspan = np.arange(10) * mc.dt
                p_avg_slope = np.polyfit(tspan, p_avg_ss, 1)[0]

                if not last_id + 1 == i: # if this point is not contiguous with last, it's a new flight
                    id += 1
            
                delta_T.append(T - all_thrust[-2])
                old_thrust.append(all_thrust[-2])
                p_top.append(fd.pwm_motors[i-mc.nmpc_delay][0])
                p_bottom.append(fd.pwm_motors[i-mc.nmpc_delay][1])
                p_avg.append(fd.control_data[i-mc.nmpc_delay][2])
                thrust.append(T)
                voltage.append(v)
                acceleration_z.append(a_z_raw)
                time.append(i)
                p_range.append(p_avg_range)
                p_slope.append(p_avg_slope)
                flight_id.append(id)
                last_id = i


p_top = np.array(p_top)
p_bottom = np.array(p_bottom)
thrust = np.array(thrust)
voltage = np.array(voltage)
delta_T = np.array(delta_T)
p_avg = np.array(p_avg)
slope = np.array(p_slope)
p_range = np.array(p_range)


##########################################################################
# now we try to figure out what the lag factor is


from scipy.optimize import minimize_scalar

steps = 5
def T_ss(p, v, coef):
    a, b, c, d, e, f = coef
    return (a + b*p + c*v + d*p**2 + e*v**2 + f*p*v)


max_steps = 10

valid_starts = []

for k in range(len(thrust) - max_steps):
    if flight_id[k] == flight_id[k + max_steps]:
        valid_starts.append(k)



def loss(beta, p_avg, voltage, thrust, coeffs, steps, valid_starts):
    errors = []

    for k in valid_starts:
        T = thrust[k]

        for j in range(k, k + steps):
            Tss = T_ss(p_avg[j], voltage[j], coeffs)
            T = T + beta * (Tss - T)

        errors.append(T - thrust[k + steps])

    return np.mean(np.square(errors))


# def count_windows(flight_id, steps):
#     count = 0

#     for k in range(len(flight_id) - steps):
#         if flight_id[k] == flight_id[k + steps]:
#             count += 1

#     return count




for s in range(5,11):
    used = []

    # for k in range(len(flight_id) - s):
    #     if flight_id[k] == flight_id[k + s]:
    #         used.append(flight_id[k])

    # print(
    #     "steps:", s,
    #     "windows:", len(used),
    #     "segments:", sorted(set(used))
    # )

    result = minimize_scalar(
        loss,
        bounds=(0.0, 1.0),
        method='bounded',
        args=(p_avg, voltage, thrust, coeffs, s, valid_starts)
    )


    beta = result.x
    print("time steps ", s)
    # print("beta =", beta)

    dt = 0.02
    tau = -dt / np.log(1 - beta)

    # print("tau =", tau, '\n')

    betas = np.linspace(0.001, 0.999, 5000)

    losses = np.array([
        loss(beta, p_avg, voltage, thrust, coeffs, s, valid_starts)
        for beta in betas
    ])

    i = np.argmin(losses)

    beta_best = betas[i]
    tau_best = -0.02 / np.log(1 - beta_best)

    print("beta =", beta_best)
    print("tau =", tau_best)
    print("MSE =", losses[i])

    betas = np.linspace(0, 1, 101)

    losses = [
        loss(beta, p_avg, voltage, thrust, coeffs, s, valid_starts)
        for beta in betas
    ]

    plt.plot(betas, losses)
    plt.xlabel("beta")
    plt.ylabel(str(s) + "-step MSE")
    plt.show()

T_pred = np.zeros(len(thrust))
T_pred[0] = thrust[0]
beta = 0.1

for k in range(1, len(thrust) - 1):
    if flight_id[k] == flight_id[k-1]:
        T_pred[k] = thrust[k]
    Tss = T_ss(p_avg[k], voltage[k], coeffs)

    T_pred[k+1] = (
        T_pred[k]
        + beta * (Tss - T_pred[k])
    )

fig = plt.figure()
# plot predicted thrust vs. flight data thrust
plt.scatter(thrust, T_pred, c=voltage, cmap='turbo', s=5)
plt.colorbar(label="Voltage (V)")

lo = min(thrust.min(), T_pred.min())
hi = max(thrust.max(), T_pred.max())

plt.plot([lo, hi], [lo, hi], 'k--')

plt.xlabel("Measured thrust (N)")
plt.ylabel("Predicted thrust (N)")
plt.axis("equal")

# note_text = (
#     "Here we fit voltage and p_avg independently with 5 coefficients but its quite a bit worse than the 6 coefficients"
# )

# note_text = textwrap.fill(note_text, width=70)

# # Reserve room for the note
# fig.subplots_adjust(bottom=0.20)

# fig.text(
#     0.1, 0.06,
#     note_text,
#     ha='left',
#     va='top',
#     fontsize=10
# )

plt.show()

