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

def plot_state_for_comparison(tspan, data, title, plot_no):

    plt.figure(figsize=(5,6))
    plt.rcParams['ytick.labelsize'] = 4 
    plt.rcParams['xtick.labelsize'] = 4

    plt.figure(plot_no)
    plt.title(title)
    plt.subplot(4, 1, 1)
    for i in range(3):
        plt.plot(tspan, data[:,i])
    plt.ylabel('$x$')

    plt.subplot(4, 1, 2)
    for i in range(3):
        plt.plot(tspan, data[:,i+3])
    plt.ylabel('$v$')

    plt.subplot(4, 1, 3)
    for i in range(4):
        plt.plot(tspan, data[:,i+6])
    plt.ylabel('$q$')

    plt.subplot(4, 1, 4)
    for i in range(3):
        plt.plot(tspan, data[:,i+10])
    plt.ylabel('$w$')

    plt.xlabel('Time (sec)')

def plot_time_comparison(tspan, d1, d2, d3, title, plot_no):
    plt.figure(figsize=(6,3))
    plt.figure(plot_no)
    plt.plot(tspan, d1, label='OC', marker='+', linestyle='None', markersize=4)
    plt.plot(tspan, d2, label='MS', marker='^', linestyle='None', markersize=4)
    plt.plot(tspan, d3, label='CPS', marker='.', linestyle='None', markersize=4)
    plt.ylabel('CPU Time (sec)')
    plt.legend(loc='lower right')
    plt.savefig("documents/time" + title  + ".pdf", format="pdf", bbox_inches="tight")



def plot_control_for_comparison(tspan, control, title, plot_no):

    plt.figure(figsize=(5,6))
    plt.rcParams['ytick.labelsize'] = 4 
    plt.rcParams['xtick.labelsize'] = 4

    plt.figure(plot_no)

    plt.title(title)
    

    # gimbal angles
    plt.subplot(3, 1, 1)
    for i in range(2):
        plt.plot(tspan, control[:,i])
    plt.ylabel('$\\theta$')

    plt.subplot(3, 1, 2)
    # average thrust
    plt.plot(tspan, control[:,2])
    plt.ylabel('$\\overline{P}$')

    plt.subplot(3, 1, 3)
    # delta thrust
    plt.plot(tspan, control[:,3])
    plt.ylabel('$\\Delta P$')

    plt.xlabel('Time')



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



def plot_state_for_paper(tspan, data, title, plot_no):

    plt.figure(figsize=(6,8))
    # plt.rcParams['ytick.labelsize'] = 10 
    # plt.rcParams['xtick.labelsize'] = 10

    plt.figure(plot_no)
    plt.subplot(4, 1, 1)
    plt.ylim(-0.5, 0.5)
    for i in range(3):
        plt.plot(tspan, data[:,i])
    plt.ylabel('$x$')

    plt.subplot(4, 1, 2)
    # plt.yticks(np.arange(-0.01, 0.02, step=0.005))
    for i in range(3):
        plt.plot(tspan, data[:,i+3])
    plt.ylabel('$v$')

    plt.subplot(4, 1, 3)
    for i in range(4):
        plt.plot(tspan, data[:,i+6])
    plt.ylabel('$q$')

    plt.subplot(4, 1, 4)
    # plt.yticks(np.arange(-0.2, 0.2, step=0.05))
    plt.ylim(-0.02, 0.02)

    for i in range(3):
        plt.plot(tspan, data[:,i+10])
    plt.ylabel('$w$')

    plt.xlabel('Time (sec)')
    plt.savefig("documents/state" + title + str(plot_no) + ".pdf", format="pdf", bbox_inches="tight")





def plot_control_for_paper(tspan, control, title, plot_no):

    plt.figure(figsize=(6,6))
    # plt.rcParams['ytick.labelsize'] = 4 
    # plt.rcParams['xtick.labelsize'] = 4

    plt.figure(plot_no)
    
    
    # gimbal angles
    plt.subplot(3, 1, 1)
    # plt.ylim(-0.05, 0.05)
    for i in range(2):
        plt.plot(tspan, control[:,i])
    plt.ylabel('$\\theta$')

    plt.subplot(3, 1, 2)
    # average thrust
    plt.plot(tspan, control[:,2])
    plt.ylabel('$\\overline{P}$')

    plt.subplot(3, 1, 3)
    plt.ylim(-0.02, 0.02)
    # delta thrust
    plt.plot(tspan, control[:,3])
    plt.ylabel('$\\Delta P$')


    plt.xlabel('Time (sec)')
    plt.savefig("documents/control" + title + str(plot_no) + ".pdf", format="pdf", bbox_inches="tight")
