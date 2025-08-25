import numpy as np
import matplotlib.pyplot as plt

def plot_state(tspan, data):
    fig, axs = plt.subplots(4)
    fig.set_figheight(8)
    fig.suptitle("State")

    for i in range(3):
        axs[0].plot(tspan, data[:,i])
    axs[0].set_ylabel('$x$')
    for i in range(3):
        axs[1].plot(tspan, data[:,i+3])
    axs[1].set_ylabel('$v$')
    for i in range(4):
        axs[2].plot(tspan, data[:,i+6])
    axs[2].set_ylabel('$q$')
    for i in range(3):
        axs[3].plot(tspan, data[:,i+10])
    axs[3].set_ylabel('$w$')

    plt.xlabel('Time')
    plt.show()


def plot_control(tspan, control):
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['xtick.labelsize'] = 8
    fig, axs = plt.subplots(3)
    fig.set_figheight(8)
    fig.suptitle("Control")


    # gimbal angles
    for i in range(2):
        axs[0].plot(tspan, control[:,i])
    axs[0].set_ylabel('$\\theta$')
   
    # average thrust
    axs[1].plot(tspan, control[:,2])
    axs[1].set_ylabel('$\\overline{P}$')

    # delta thrust
    axs[2].plot(tspan, control[:,3])
    axs[2].set_ylabel('$\\Delta P$')

    plt.xlabel('Time')
    plt.show()



def plot_pwm(tspan, pwm_servos, pwm_motors):
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['xtick.labelsize'] = 8
    fig, axs = plt.subplots(3)
    fig.set_figheight(8)
    fig.suptitle("Control PWM")

    # pwm gimbal angles
    for i in range(2):
        axs[0].plot(tspan, pwm_servos[:,i])
    axs[0].set_ylabel('$\\theta$')

    # pwm thrust
    axs[1].plot(tspan, pwm_motors[:,0])
    axs[1].set_ylabel('$\\overline{P}$')
    axs[2].plot(tspan, pwm_motors[:,1])
    axs[2].set_ylabel('$\\Delta P$')

    plt.xlabel('Time')
    plt.show()

def plot_state_control(axs, tspan, data, control):

    for i in range(3):
        axs[0].plot(tspan, data[:,i])
    axs[0].set_ylabel('$x$')
    for i in range(3):
        axs[1].plot(tspan, data[:,i+3])
    axs[1].set_ylabel('$v$')
    for i in range(4):
        axs[2].plot(tspan, data[:,i+6])
    axs[2].set_ylabel('$q$')
    for i in range(3):
        axs[3].plot(tspan, data[:,i+10])
    axs[3].set_ylabel('$w$')


    # gimbal angles
    for i in range(2):
        axs[4].plot(tspan, control[:,i])
    axs[4].set_ylabel('$\\theta$')
   
    # average thrust
    axs[5].plot(tspan, control[:,2])
    axs[5].set_ylabel('$\\overline{P}$')

    min_val = np.min(control[:,3])
    max_val = np.max(control[:,3])

    # delta thrust
    axs[6].plot(tspan, control[:,3])
    axs[6].set_ylabel('$\\Delta P$')


    axs[6].set_ylim(min(min_val, -0.01), max(max_val, 0.01))