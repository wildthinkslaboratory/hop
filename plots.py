import numpy as np
import matplotlib.pyplot as plt



def plot_state(tspan, data, title='State'):
    fig, axs = plt.subplots(4)
    fig.set_figheight(8)
    fig.suptitle(title)

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
    # plt.show()


def plot_control(tspan, control, title='Control'):
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['xtick.labelsize'] = 8
    fig, axs = plt.subplots(3)
    fig.set_figheight(8)
    fig.suptitle(title)


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
    # plt.show()

def plot_attitude(tspan, attitude, title='Attitude'):
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['xtick.labelsize'] = 8
    fig, axs = plt.subplots(3)
    fig.set_figheight(8)
    fig.suptitle(title)


    # x angle

    axs[0].plot(tspan, attitude[:,0])
    axs[0].set_ylabel('$x$')
   
    # y angle
    axs[1].plot(tspan, attitude[:,1])
    axs[1].set_ylabel('$y$')

    # total angle
    axs[2].plot(tspan, attitude[:,2])
    axs[2].set_ylabel('tilt')

    plt.xlabel('Time')
    # plt.show()

def plot_parameters(tspan, parameters, title='Parameters'):
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['xtick.labelsize'] = 8
    fig, axs = plt.subplots(2)
    fig.set_figheight(8)
    fig.suptitle(title)

    for i in range(3):
        axs[0].plot(tspan, parameters[:,i])
    axs[0].set_ylabel('$x$')
 

    axs[1].plot(tspan, parameters[:,3])
    axs[1].set_ylabel('V')
   


    plt.xlabel('Time')
    # plt.show()


def plot_pwm(tspan, pwm_servos, pwm_motors, title='pwm'):
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['xtick.labelsize'] = 8
    fig, axs = plt.subplots(3)
    fig.set_figheight(8)
    fig.suptitle(title)

    # pwm gimbal angles
    for i in range(2):
        axs[0].plot(tspan, pwm_servos[:,i])
    axs[0].set_ylabel('$\\theta$')

    # pwm thrust
    axs[1].plot(tspan, pwm_motors[:,0])
    axs[1].set_ylabel('top motor')
    axs[2].plot(tspan, pwm_motors[:,1])
    axs[2].set_ylabel('lower motor')

    plt.xlabel('Time')
    plt.show()


# this plots state and control together so you can see
# then all in a stack.
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


##########################################################################
#
# These next set of plots were used when we were running simulations and 
# comparing different methods.
#
##########################################################################

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


##########################################################################
#
# These next set of plots were used to generate plots for the simulations
# in our paper.
#
##########################################################################

use_limits = False

state_lims = {
    '45dz': [[-0.5, 0.5], [-0.005, 0.015], [-0.2, 1.2], [-0.2, 0.05]],
    'x1z1vx': [[-0.6, 1.6], [-1.0, 0.8], [-0.2, 1.2], [-1.0, 0.6]],
    'y115dx': [[-1.5, 1.2], [-3, 1.2], [-0.2, 1.2], [-1.5, 0.5]],
    'hover': [[-0.5, 0.5], [-0.5, 0.5], [-0.2, 1.2], [-0.5, 0.5]]

}
def plot_state_for_paper(tspan, data, title, plot_no):

    plt.figure(figsize=(6,8))
    # plt.rcParams['ytick.labelsize'] = 10 
    # plt.rcParams['xtick.labelsize'] = 10

    plt.figure(plot_no)

    plt.subplot(4, 1, 1)
    if use_limits:
        plt.ylim(state_lims[title][0])
    for i in range(3):
        plt.plot(tspan, data[:,i])
    plt.ylabel('$x$')


    plt.subplot(4, 1, 2)
    if use_limits:
        plt.ylim(state_lims[title][1])
    for i in range(3):
        plt.plot(tspan, data[:,i+3])
    plt.ylabel('$v$')

    plt.subplot(4, 1, 3)
    if use_limits:
        plt.ylim(state_lims[title][2])
    for i in range(4):
        plt.plot(tspan, data[:,i+6])
    plt.ylabel('$q$')

    plt.subplot(4, 1, 4)
    if use_limits:
        plt.ylim(state_lims[title][3])
    for i in range(3):
        plt.plot(tspan, data[:,i+10])
    plt.ylabel('$w$')

    plt.xlabel('Time (sec)')
    plt.savefig("state" + title + str(plot_no) + ".pdf", format="pdf", bbox_inches="tight")


control_lims = {
    '45dz': [[-0.5, 0.5], [2.4, 2.8], [-1.8, 0.5]],
    'x1z1vx': [[-3.0, 7.0], [2.0, 2.8], [-0.4, 0.4]],

    'y115dx': [[-10.0, 25.0], [1.0, 8.0], [-0.25, 0.25]],

    'hover': [[-0.2, 0.2], [2.3, 3.8], [-0.02, 0.02]]
}

def plot_control_for_paper(tspan, control, title, plot_no):

    plt.figure(figsize=(6,6))
    # plt.rcParams['ytick.labelsize'] = 4 
    # plt.rcParams['xtick.labelsize'] = 4

    plt.figure(plot_no)
    
    
    # gimbal angles
    plt.subplot(3, 1, 1)
    if use_limits:
        plt.ylim(control_lims[title][0])
    for i in range(2):
        plt.plot(tspan, control[:,i])
    plt.ylabel('$\\theta$')

    # average thrust
    plt.subplot(3, 1, 2)
    if use_limits:
        plt.ylim(control_lims[title][1])
    plt.plot(tspan, control[:,2])
    plt.ylabel('$\\overline{P}$')

    # delta thrust
    plt.subplot(3, 1, 3)
    if use_limits:
        plt.ylim(control_lims[title][2])
    plt.plot(tspan, control[:,3])
    plt.ylabel('$\\Delta P$')


    plt.xlabel('Time (sec)')
    plt.savefig("control" + title + str(plot_no) + ".pdf", format="pdf", bbox_inches="tight")

markers = { 'oc': '+', 'cps': '.', 'ms':'^'}
def plot_comparison(tspan, data, title, plot_no, ylab):
    plt.figure(figsize=(6,3))
    plt.figure(plot_no)
    keys = data.keys()
    for key in keys:
        if not len(data[key]) == 0:
            plt.plot(tspan, data[key], label=key, marker=markers[key], linestyle='None', markersize=4)
    plt.ylabel(ylab)
    plt.legend(loc='lower right')
    plt.savefig("time" + title  + ".pdf", format="pdf", bbox_inches="tight")

def plot_state_for_sensitivity(tspan, data, title, plot_no):

    plt.figure(figsize=(6,8))
    plt.figure(plot_no)

    plt.subplot(4, 1, 1)
    plt.ylim([-1, 1.5])
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
    plt.savefig("state" + title + str(plot_no) + ".pdf", format="pdf", bbox_inches="tight")



def plot_control_for_sensitivity(tspan, control, title, plot_no):

    plt.figure(figsize=(6,6))
    plt.figure(plot_no)
    
    
    # gimbal angles
    plt.subplot(3, 1, 1)

    plt.ylim([-5,5])
    for i in range(2):
        plt.plot(tspan, control[:,i])
    plt.ylabel('$\\theta$')

    # average thrust
    plt.subplot(3, 1, 2)
    plt.ylim([0,1])
    plt.plot(tspan, control[:,2])
    plt.ylabel('$\\overline{P}$')

    # delta thrust
    plt.subplot(3, 1, 3)
    plt.ylim([-0.5, 0.5])
    plt.plot(tspan, control[:,3])
    plt.ylabel('$\\Delta P$')


    plt.xlabel('Time (sec)')
    plt.savefig("control" + title + str(plot_no) + ".pdf", format="pdf", bbox_inches="tight")
