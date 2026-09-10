from hop.constants import Constants
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from flight_analysis_tools.flight_data import FlightData
from hop.utilities import quaternion_to_angle, estimate_thrust_from_state
from scipy.optimize import least_squares



def get_az(vz, dt):
    t = np.arange(5) * dt
    slope, _ = np.polyfit(t, vz, 1)
    return slope


# put all the flight logs that we want to analyze in this directory
directory = Path("thrust_files")




# for a in range(10):
#     alpha = 0.9 + a / 100.0
#     a_z_prev = 0.0

thrust = []
voltage = []
p_top = []
p_bottom = []
all_thrust = []
old_thrust = []
delta_T = []
time = []

ss_window = 20
max_p_range = 0.02
max_p_slope = 0.05 
ss_thrust = []
ss_voltage = []
ss_p_avg = []
ss_p_diff = []
ss_old_thrust = []
ss_time = []
ss_v_z = []
ss_z = []
ss_current = []
ss_flight_no = []
id = 0
last_id = 0

for file in directory.iterdir():
    id += 1 # starting a new contiguous set of data points
    mc = Constants()
    fd = FlightData(file)
    mc.update_from_dictionary(fd.constants)

    for i in range(len(fd.state_data) - 2):# min(len(fd.state_data) - 2,100)):#len(fd.state_data) - 2):
        if i > 5:
            # estimate the thrust in Newtons
            # v = fd.parameters[i][3]  # read the filtered voltage value from the parameters
            v = fd.raw_voltage[i]
            # a_z_raw = get_az(fd.state_data[i-1:i, 5], mc.dt)

            # # we account for any rotation of the drone
            # # x_theta, y_theta, theta = quaternion_to_angle(fd.state_data[i][6:10])
            # T =  mc.m * (-mc.gz + a_z_raw) / np.cos(theta * np.pi / 180.0)
            T = estimate_thrust_from_state(fd.state_data[i-1:i+1, 5], mc.m, fd.state_data[i][6:10], mc.dt)
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
            if i > (ss_window + mc.nmpc_delay) and r_xy < 0.1 and z > 0.68: # and abs(fd.control_data[i][3]) <= 0.01 and v < 21.0: 

                delta_T.append(T - all_thrust[-2])
                old_thrust.append(all_thrust[-2])
                p_top.append(fd.pwm_motors[i-mc.nmpc_delay][0])
                p_bottom.append(fd.pwm_motors[i-mc.nmpc_delay][1])
                thrust.append(T)
                voltage.append(v)
                time.append(i)

                p_avg_ss = fd.control_data[i-mc.nmpc_delay - ss_window: i-mc.nmpc_delay, 2]
                p_avg_range = np.max(p_avg_ss) - np.min(p_avg_ss)
                tspan = np.arange(ss_window) * mc.dt
                p_avg_slope = np.polyfit(tspan, p_avg_ss, 1)[0]

                    # if p_avg_range < max_p_range and abs(p_avg_slope) < max_p_slope:
                    # if not last_id + 1 == i: # if this point is not contiguous with last, it's a new flight
                    #     id += 1
                ss_thrust.append(T)
                ss_voltage.append(voltage[-1])
                ss_p_avg.append(fd.control_data[i-mc.nmpc_delay][2])
                ss_p_diff.append(fd.control_data[i-mc.nmpc_delay][3])
                ss_old_thrust.append(old_thrust[-1])
                ss_time.append(float(i))
                ss_v_z.append(fd.state_data[i][5])
                ss_current.append(fd.current[i])
                ss_flight_no.append(float(id))
                ss_z.append(fd.state_data[i][2])

p_top = np.array(p_top)
p_bottom = np.array(p_bottom)
thrust = np.array(thrust)
voltage = np.array(voltage)
delta_T = np.array(delta_T)
p_avg = (p_top + p_bottom) / 2

ss_old_thrust = np.array(ss_old_thrust)
ss_p_avg = np.array(ss_p_avg)
ss_p_diff = np.array(ss_p_diff)
ss_thrust = np.array(ss_thrust)
ss_voltage = np.array(ss_voltage)
ss_v_z = np.array(ss_v_z)
ss_current = np.array(ss_current)
ss_z = np.array(ss_z)
ss_flight_no = np.array(ss_flight_no)


p_diff = (p_top - p_bottom) / 2
p_diff_abs = abs((p_top - p_bottom) / 2)
p_avg_scaled = p_avg * 25.0 / voltage


##########################################################################

plt.scatter(ss_p_avg, ss_thrust, c=ss_voltage, cmap='turbo', s=5)
plt.axhline(15.86, color='k', linestyle='--')
plt.colorbar(label="steady state voltage (V)")

plt.ylabel("steady state thrust (N)")
plt.xlabel("steady state PWM average")
plt.show()

##########################################################################

plt.scatter(ss_p_avg, ss_thrust, c=ss_p_diff, cmap='turbo', s=5)
plt.axhline(15.86, color='k', linestyle='--')
plt.colorbar(label="steady state PWM diff")

plt.ylabel("steady state thrust (N)")
plt.xlabel("steady state PWM average")
plt.show()


##########################################################################

plt.scatter(ss_p_avg, ss_thrust, c=ss_current, cmap='turbo', s=5)
plt.axhline(15.86, color='k', linestyle='--')
plt.colorbar(label="steady state current")

plt.ylabel("steady state thrust (N)")
plt.xlabel("steady state PWM average")
plt.show()


##########################################################################

plt.scatter(ss_p_avg, ss_thrust, c=ss_v_z, cmap='turbo', s=5)
plt.axhline(15.86, color='k', linestyle='--')
plt.colorbar(label="steady state v_z")

plt.ylabel("steady state thrust (N)")
plt.xlabel("steady state PWM average")
plt.show()


##########################################################################

plt.scatter(ss_p_avg, ss_thrust, c=ss_time, cmap='turbo', s=5)
plt.axhline(15.86, color='k', linestyle='--')
plt.colorbar(label="steady state time index")

plt.ylabel("steady state thrust (N)")
plt.xlabel("steady state PWM average")
plt.show()

##########################################################################

plt.scatter(ss_p_avg, ss_thrust, c=ss_flight_no, cmap='turbo', s=5)
plt.axhline(15.86, color='k', linestyle='--')
plt.colorbar(label="steady state flight no")

plt.ylabel("steady state thrust (N)")
plt.xlabel("steady state PWM average")
plt.show()


##########################################################################

plt.scatter(ss_p_avg, ss_thrust, c=ss_z, cmap='turbo', s=5)
plt.axhline(15.86, color='k', linestyle='--')
plt.colorbar(label="steady state z")

plt.ylabel("steady state thrust (N)")
plt.xlabel("steady state PWM average")
plt.show()

##########################################################################

plt.scatter(ss_p_avg, ss_thrust, c=ss_old_thrust, cmap='turbo', s=5)
plt.axhline(15.86, color='k', linestyle='--')
plt.colorbar(label="steady state previous thrust")

plt.ylabel("steady state thrust (N)")
plt.xlabel("steady state PWM average")
plt.show()

##########################################################################
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(
    p_top,
    p_bottom,
    thrust,
    c=old_thrust,
    cmap='turbo',
    s=8
)

ax.set_xlabel("Top PWM")
ax.set_ylabel("Bottom PWM")
ax.set_zlabel("Thrust (N)")
plt.colorbar(sc, label="Previous Thrust")
plt.show()

#########################################################################

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(
    p_avg,
    voltage,
    delta_T,
    c=time,
    cmap='turbo',
    s=8
)

ax.set_xlabel("Average PWM")
ax.set_ylabel("voltage (V)")
ax.set_zlabel("Delta T (N)")
plt.colorbar(sc, label="Time (i)")
plt.show()

##########################################################################

##########################################################################
import textwrap

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

sc = ax.scatter(
    p_avg,
    old_thrust,
    thrust,
    c=time,
    cmap='turbo',
    s=8
)

ax.set_xlabel("PWM avg")
ax.set_ylabel("Previous Thrust")
ax.set_zlabel("Thrust (N)")

plt.colorbar(sc, label="voltage")

note_text = (
    "You can see that previous thrust is the best predictor "
    "how much thrust will result from a given PWM and voltage."
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

# we fit the data as a quadratic with pwm top, pwm bottom and voltage
# all being independent of each other

# X = np.column_stack([
#     np.ones_like(p_top),

#     old_thrust,
#     p_top,
#     p_bottom,
#     voltage,

#     p_top**2,
#     p_bottom**2,
#     voltage**2,

#     p_top * p_bottom,
#     p_top * voltage,
#     p_bottom * voltage
# ])

# coeffs, *_ = np.linalg.lstsq(X, thrust, rcond=None)

# predicted_thrust = X @ coeffs
# error = thrust - predicted_thrust

# rmse = np.sqrt(np.mean((error)**2))
# r2 = 1 - np.sum((error)**2) / \
#         np.sum((thrust - np.mean(thrust))**2)

# print("coefficients:")
# for i, c in enumerate(coeffs):
#     print(f"c{i} = {c:.8f}")

# print("RMSE:", rmse)
# print("R²:", r2)
# print("thrust std:", np.std(thrust))


##########################################################################

# # plot predicted thrust vs. flight data thrust
# plt.scatter(thrust, predicted_thrust, c=old_thrust, cmap='turbo', s=5)
# plt.colorbar(label="T_[k-window]")

# lo = min(thrust.min(), predicted_thrust.min())
# hi = max(thrust.max(), predicted_thrust.max())

# plt.plot([lo, hi], [lo, hi], 'k--')

# plt.xlabel("Measured thrust (N)")
# plt.ylabel("Predicted thrust (N)")
# plt.axis("equal")
# plt.show()

# X = np.column_stack([
#     np.ones_like(p_top),

#     old_thrust,
#     p_top,
#     p_bottom,
#     voltage,

#     p_top**2,
#     p_bottom**2,
#     voltage**2,

#     p_top * p_bottom,
#     p_top * voltage,
#     p_bottom * voltage
# ])

# coeffs, *_ = np.linalg.lstsq(X, delta_T, rcond=None)

# predicted_thrust = X @ coeffs
# error = delta_T - predicted_thrust

# rmse = np.sqrt(np.mean((error)**2))
# r2 = 1 - np.sum((error)**2) / \
#         np.sum((delta_T - np.mean(delta_T))**2)

# print("coefficients:")
# for i, c in enumerate(coeffs):
#     print(f"c{i} = {c:.8f}")

# print("RMSE:", rmse)
# print("R²:", r2)
# print("thrust std:", np.std(delta_T))

# print("R^2", r2, " alpha ", alpha)
##########################################################################

# # plot predicted thrust vs. flight data thrust
# plt.scatter(delta_T, predicted_thrust, c=voltage, cmap='turbo', s=5)
# plt.colorbar(label="voltage (V)")

# lo = min(delta_T.min(), predicted_thrust.min())
# hi = max(delta_T.max(), predicted_thrust.max())

# plt.plot([lo, hi], [lo, hi], 'k--')

# plt.xlabel("Measured delta thrust (N)")
# plt.ylabel("Predicted delta thrust (N)")
# plt.axis("equal")
# plt.show()


##########################################################################

# we fit the data as a quadratic with pwm top, pwm bottom and voltage
# all being independent of each other

# X = np.column_stack([
#     np.ones_like(p_top),
#     old_thrust,
# ])

# coeffs, *_ = np.linalg.lstsq(X, thrust, rcond=None)

# predicted_thrust = X @ coeffs
# error = thrust - predicted_thrust

# rmse = np.sqrt(np.mean((error)**2))
# r2 = 1 - np.sum((error)**2) / \
#         np.sum((thrust - np.mean(thrust))**2)

# print("coefficients:")
# for i, c in enumerate(coeffs):
#     print(f"c{i} = {c:.8f}")

# print("RMSE:", rmse)
# print("R²:", r2)
# print("thrust std:", np.std(thrust))



# print("R^2", r2, " alpha ", alpha)
##########################################################################

# plot predicted thrust vs. flight data thrust
# plt.scatter(thrust, predicted_thrust, c=old_thrust, cmap='turbo', s=5)
# plt.colorbar(label="T_[k-window]")

# lo = min(thrust.min(), predicted_thrust.min())
# hi = max(thrust.max(), predicted_thrust.max())

# plt.plot([lo, hi], [lo, hi], 'k--')

# plt.xlabel("Measured thrust (N)")
# plt.ylabel("Predicted thrust (N)")
# plt.axis("equal")
# plt.show()



# let's try a simpler model and see how it compares

# A = np.column_stack((p_avg_scaled**2, p_avg_scaled, np.ones_like(p_top)))
# coeffs_2, _, _, _ = np.linalg.lstsq(A, thrust, rcond=None)

# a, b, c = coeffs_2

# print('a: ', a / 9.81)
# print('b: ', b / 9.81)
# print('c: ', c / 9.81)

# predicted_thrust_2 = A @ coeffs_2
# error_2 = thrust - predicted_thrust_2

# rmse = np.sqrt(np.mean((error_2)**2))
# r2 = 1 - np.sum((error_2)**2) / \
#         np.sum((thrust - np.mean(thrust))**2)

# print("coefficients:")
# for i, c in enumerate(coeffs_2):
#     print(f"c{i} = {c:.8f}")

# print("RMSE:", rmse)
# print("R²:", r2)
# print("thrust std:", np.std(thrust))


##########################################################################

# plt.figure(3)
# # plot predicted thrust vs. flight data thrust
# plt.scatter(thrust, predicted_thrust_2, c=voltage, cmap='turbo', s=5)
# plt.colorbar(label="Voltage (V)")

# lo = min(thrust.min(), predicted_thrust_2.min())
# hi = max(thrust.max(), predicted_thrust_2.max())

# plt.plot([lo, hi], [lo, hi], 'k--')

# plt.xlabel("Measured thrust (N)")
# plt.ylabel("Predicted thrust simple (N)")
# plt.axis("equal")
# plt.show()


# #########################################################################
# # look for patterns in the errors. Do we have higher errors in
# # any subset of data? High voltage, high p_diff?

# plt.scatter(p_diff, error, c=voltage, cmap='turbo', s=8)
# plt.axhline(0, color='k', linestyle='--')
# plt.xlabel("Diff PWM")
# plt.ylabel("Thrust residual (N)")
# plt.colorbar(label="Voltage (V)")
# plt.show()



# ##########################################################################

# # look at relationship between average PWM and voltage
# plt.scatter(p_avg_scaled, acceleration_z, c=voltage, cmap='turbo', s=8)
# plt.axhline(0, color='k', linestyle='--')
# plt.xlabel("PWM average")
# plt.ylabel("Vertical acceleration")
# plt.colorbar(label="Voltage (V)")
# plt.show()


# ##########################################################################

# # look at relationship between average PWM and voltage
# plt.scatter(thrust, acceleration_z, c=voltage, cmap='turbo', s=8)
# plt.axhline(0, color='k', linestyle='--')
# plt.xlabel("thrust (N)")
# plt.ylabel("Vertical acceleration")
# plt.colorbar(label="Voltage (V)")
# plt.show()



# ##########################################################################