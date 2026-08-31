from hop.constants import Constants
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from flight_analysis_tools.flight_data import FlightData
from hop.utilities import quaternion_to_angle
from scipy.optimize import least_squares



# put all the flight logs that we want to analyze in this directory
directory = Path("thrust_files")




# for a in range(10):
#     alpha = 0.9 + a / 100.0
#     a_z_prev = 0.0

# We're trying to understand how the PWM of the top and bottom
# motors and voltage map to the generated thrust.
thrust = []
voltage = []
p_top = []
p_bottom = []
acceleration_z = []
all_thrust = []
old_thrust = []
delta_T = []

time = []
def get_az(vz, dt):
    t = np.arange(5) * dt
    slope, _ = np.polyfit(t, vz, 1)
    return slope


for file in directory.iterdir():
    # print(file)
    mc = Constants()
    fd = FlightData(file)
    mc.update_from_dictionary(fd.constants)

    for i in range(len(fd.state_data) - 2):# min(len(fd.state_data) - 2,100)):#len(fd.state_data) - 2):
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


            if i > 6 and abs(fd.control_data[i][3]) <= 0.08 and r_xy < 0.1 and z > 0.7:

                delta_T.append(T - all_thrust[-2])
                old_thrust.append(all_thrust[-2])
                p_top.append(fd.pwm_motors[i-mc.nmpc_delay][0])
                p_bottom.append(fd.pwm_motors[i-mc.nmpc_delay][1])
                thrust.append(T)
                voltage.append(v)
                acceleration_z.append(a_z_raw)
                time.append(i)


p_top = np.array(p_top)
p_bottom = np.array(p_bottom)
thrust = np.array(thrust)
voltage = np.array(voltage)
delta_T = np.array(delta_T)
p_avg = (p_top + p_bottom) / 2

p_diff = (p_top - p_bottom) / 2
p_diff_abs = abs((p_top - p_bottom) / 2)

p_avg_scaled = p_avg * 25.0 / voltage
##########################################################################
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# sc = ax.scatter(
#     p_top,
#     p_bottom,
#     thrust,
#     c=voltage,
#     cmap='turbo',
#     s=8
# )

# ax.set_xlabel("Top PWM")
# ax.set_ylabel("Bottom PWM")
# ax.set_zlabel("Thrust (N)")
# plt.colorbar(sc, label="Voltage (V)")
# plt.show()

#########################################################################



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# sc = ax.scatter(
#     p_avg,
#     p_diff,
#     thrust,
#     c=voltage,
#     cmap='turbo',
#     s=8
# )

# ax.set_xlabel("Average PWM")
# ax.set_ylabel("Differential PWM")
# ax.set_zlabel("Thrust (N)")
# plt.colorbar(sc, label="Voltage (V)")
# plt.show()

##########################################################################

# look at relationship between average PWM and voltage
plt.scatter(p_avg, thrust, c=old_thrust, cmap='turbo', s=8)
plt.xlabel("PWM average")
plt.ylabel("Thrust (N)")
plt.colorbar(label="pre windo Thrust (N)")
plt.show()


# ##########################################################################

# # look at relationship between average PWM and differential thrust
# plt.scatter(p_avg, thrust, c=p_diff_abs, cmap='turbo', s=8)
# plt.xlabel("PWM average")
# plt.ylabel("Thrust (N)")
# plt.colorbar(label="PWM differential")
# plt.show()

##########################################################################

# we fit the data as a quadratic with pwm top, pwm bottom and voltage
# all being independent of each other

X = np.column_stack([
    np.ones_like(p_top),

    old_thrust,
    p_top,
    p_bottom,
    voltage,

    p_top**2,
    p_bottom**2,
    voltage**2,

    p_top * p_bottom,
    p_top * voltage,
    p_bottom * voltage
])

coeffs, *_ = np.linalg.lstsq(X, thrust, rcond=None)

predicted_thrust = X @ coeffs
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

# print("R^2", r2, " alpha ", alpha)
##########################################################################

# plot predicted thrust vs. flight data thrust
plt.scatter(thrust, predicted_thrust, c=old_thrust, cmap='turbo', s=5)
plt.colorbar(label="T_[k-window]")

lo = min(thrust.min(), predicted_thrust.min())
hi = max(thrust.max(), predicted_thrust.max())

plt.plot([lo, hi], [lo, hi], 'k--')

plt.xlabel("Measured thrust (N)")
plt.ylabel("Predicted thrust (N)")
plt.axis("equal")
plt.show()

X = np.column_stack([
    np.ones_like(p_top),

    old_thrust,
    p_top,
    p_bottom,
    voltage,

    p_top**2,
    p_bottom**2,
    voltage**2,

    p_top * p_bottom,
    p_top * voltage,
    p_bottom * voltage
])

coeffs, *_ = np.linalg.lstsq(X, delta_T, rcond=None)

predicted_thrust = X @ coeffs
error = delta_T - predicted_thrust

rmse = np.sqrt(np.mean((error)**2))
r2 = 1 - np.sum((error)**2) / \
        np.sum((delta_T - np.mean(delta_T))**2)

print("coefficients:")
for i, c in enumerate(coeffs):
    print(f"c{i} = {c:.8f}")

print("RMSE:", rmse)
print("R²:", r2)
print("thrust std:", np.std(delta_T))

# print("R^2", r2, " alpha ", alpha)
##########################################################################

# plot predicted thrust vs. flight data thrust
plt.scatter(delta_T, predicted_thrust, c=voltage, cmap='turbo', s=5)
plt.colorbar(label="voltage (V)")

lo = min(delta_T.min(), predicted_thrust.min())
hi = max(delta_T.max(), predicted_thrust.max())

plt.plot([lo, hi], [lo, hi], 'k--')

plt.xlabel("Measured delta thrust (N)")
plt.ylabel("Predicted delta thrust (N)")
plt.axis("equal")
plt.show()


##########################################################################

# we fit the data as a quadratic with pwm top, pwm bottom and voltage
# all being independent of each other

X = np.column_stack([
    np.ones_like(p_top),
    old_thrust,
])

coeffs, *_ = np.linalg.lstsq(X, thrust, rcond=None)

predicted_thrust = X @ coeffs
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



# print("R^2", r2, " alpha ", alpha)
##########################################################################

# plot predicted thrust vs. flight data thrust
plt.scatter(thrust, predicted_thrust, c=old_thrust, cmap='turbo', s=5)
plt.colorbar(label="T_[k-window]")

lo = min(thrust.min(), predicted_thrust.min())
hi = max(thrust.max(), predicted_thrust.max())

plt.plot([lo, hi], [lo, hi], 'k--')

plt.xlabel("Measured thrust (N)")
plt.ylabel("Predicted thrust (N)")
plt.axis("equal")
plt.show()



# let's try a simpler model and see how it compares

A = np.column_stack((p_avg_scaled**2, p_avg_scaled, np.ones_like(p_top)))
coeffs_2, _, _, _ = np.linalg.lstsq(A, thrust, rcond=None)

a, b, c = coeffs_2

print('a: ', a / 9.81)
print('b: ', b / 9.81)
print('c: ', c / 9.81)

predicted_thrust_2 = A @ coeffs_2
error_2 = thrust - predicted_thrust_2

rmse = np.sqrt(np.mean((error_2)**2))
r2 = 1 - np.sum((error_2)**2) / \
        np.sum((thrust - np.mean(thrust))**2)

print("coefficients:")
for i, c in enumerate(coeffs_2):
    print(f"c{i} = {c:.8f}")

print("RMSE:", rmse)
print("R²:", r2)
print("thrust std:", np.std(thrust))


##########################################################################

plt.figure(3)
# plot predicted thrust vs. flight data thrust
plt.scatter(thrust, predicted_thrust_2, c=voltage, cmap='turbo', s=5)
plt.colorbar(label="Voltage (V)")

lo = min(thrust.min(), predicted_thrust_2.min())
hi = max(thrust.max(), predicted_thrust_2.max())

plt.plot([lo, hi], [lo, hi], 'k--')

plt.xlabel("Measured thrust (N)")
plt.ylabel("Predicted thrust simple (N)")
plt.axis("equal")
plt.show()


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