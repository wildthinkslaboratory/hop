#
# Plot the data logs from a drone run
#

from hop.utilities import import_data
import numpy as np
import matplotlib.pyplot as plt
from animation import RocketAnimation
from plots import plot_state, plot_control, plot_pwm
import sys

log_file_name = './plotter_logs/current.json'
if len(sys.argv) > 1:
    log_file_name = sys.argv[1]



#log = import_data('./plotter_logs/2025-12-25-15-49log.json')  
log = import_data(log_file_name)    
# constants = log['constants']
# dt = constants['dt']
dt = 0.02
data = log['run_data']

tspan = np.arange(0, len(data) * dt , dt)
print(tspan)
state_data = np.empty([len(data),13])
control_data = np.empty([len(data),4])
pwm_servos = np.empty([len(data),2])
pwm_motors = np.empty([len(data),2])
voltage = np.empty(len(data))

# collect all the data into arrays
for i, d in enumerate(data):
    state_data[i] = np.array(d['state'])
    control_data[i] = np.array(d['control'])
    pwm_servos[i] = np.array(d['pwm_servos'])
    pwm_motors[i] = np.array(d['pwm_motors'])
    voltage[i] = d['voltage']

# now all the plots
plt.figure(figsize=(6,3))

plt.figure(1)
plt.plot(tspan, voltage)

# build the plots
print(tspan.shape, state_data.shape)
plot_state(tspan, state_data)
plot_control(tspan, control_data)
plot_pwm(tspan, pwm_servos, pwm_motors)
plt.show()

# plot_pwm(tspan, pwm_servos, pwm_motors)

# show an animation for state and control
rc = RocketAnimation()
rc.animate(tspan, state_data, control_data)