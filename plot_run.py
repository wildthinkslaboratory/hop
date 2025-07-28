from hop.utilities import import_data
import numpy as np
import matplotlib.pyplot as plt
from animation import RocketAnimation

log = import_data('./plotter_logs/current.json')
constants = log['constants']
data = log['run_data']

tspan = np.arange(0, len(data) * constants['dt'], constants['dt'])
state_data = np.empty([len(data),13])
control_data = np.empty([len(data),4])
pwm_servos = np.empty([len(data),2])
pwm_motors = np.empty([len(data),2])

for i, d in enumerate(data):
    state_data[i] = np.array(d['state'])
    control_data[i] = np.array(d['control'])
    pwm_servos[i] = np.array(d['pwm_servos'])
    pwm_motors[i] = np.array(d['pwm_motors'])

# first we print out the state in one plot

fig, axs = plt.subplots(4)
fig.set_figheight(8)
fig.suptitle("Run Analysis State")

for i in range(3):
    axs[0].plot(tspan, state_data[:,i])
axs[0].set_ylabel('x')
for i in range(3):
    axs[1].plot(tspan, state_data[:,i+3])
axs[1].set_ylabel('v')
for i in range(4):
    axs[2].plot(tspan, state_data[:,i+6])
axs[2].set_ylabel('q')
for i in range(3):
    axs[3].plot(tspan, state_data[:,i+10])
axs[3].set_ylabel('w')

plt.xlabel('Time')
plt.show()

fig, axs = plt.subplots(4)
fig.set_figheight(8)
fig.suptitle("Run Analysis Control")

for i in range(2):
    axs[0].plot(tspan, control_data[:,i])
axs[0].set_ylabel('g')
for i in range(2):
    axs[1].plot(tspan, control_data[:,i+2])
axs[1].set_ylabel('t')
for i in range(2):
    axs[2].plot(tspan, pwm_servos[:,i])
axs[2].set_ylabel('pg')
for i in range(2):
    axs[3].plot(tspan, pwm_motors[:,i])
axs[3].set_ylabel('pt')

plt.xlabel('Time')
plt.show()

rc = RocketAnimation()
rc.animate(tspan, state_data, control_data)