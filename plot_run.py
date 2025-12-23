#
# Plot the data logs from a drone run
#

from hop.utilities import import_data
import numpy as np
import matplotlib.pyplot as plt
from animation import RocketAnimation
from plots import plot_state, plot_control, plot_pwm


log = import_data('./plotter_logs/current.json')    
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

# collect all the data into arrays
for i, d in enumerate(data):
    state_data[i] = np.array(d['state'])
    control_data[i] = np.array(d['control'])
    pwm_servos[i] = np.array(d['pwm_servos'])
    pwm_motors[i] = np.array(d['pwm_motors'])

# build the plots
print(tspan.shape, state_data.shape)
plot_state(tspan, state_data)
plot_control(tspan, control_data)
plot_pwm(tspan, pwm_servos, pwm_motors)

# show an animation for state and control
rc = RocketAnimation()
rc.animate(tspan, state_data, control_data)