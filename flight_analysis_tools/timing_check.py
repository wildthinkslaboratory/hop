import matplotlib.pyplot as plt
import numpy as np
from flight_analysis_tools.flight_data import FlightData


fd = FlightData()
px4_start = fd.timing_data[0][0]


fig, ax = plt.subplots(figsize=(12,6))

# np.random.seed(0)


n_cycles = len(fd.timing_data)-1
n_cycles = 20


# we map our first pi time to the time of the first pixhawk send time 
# plus a time delta for the message time.
def pi_to_pixhawk_clock(pi_time):
    px4_pi_delta = 10000 # in ms
    px4_first_send_time = fd.timing_data[0][1]
    first_pi_time = fd.timing_data[0][2]
    
    # pi time is in 10^-6 seconds and px4 is in 10^-5 seconds
    return px4_first_send_time + px4_pi_delta + (pi_time - first_pi_time) * 1000
    


for i in range(n_cycles):
    print(fd.timing_data[i])
    # first get everything into ms
    state_sample = (fd.timing_data[i][0] - px4_start) / 1000
    state_sent = (fd.timing_data[i][1] - px4_start) / 1000
    state_receive = (pi_to_pixhawk_clock(fd.timing_data[i][2]) - px4_start) / 1000
    nmpc_start = (pi_to_pixhawk_clock(fd.timing_data[i][3]) - px4_start) / 1000
    nmpc_end = (pi_to_pixhawk_clock(fd.timing_data[i][4]) - px4_start) / 1000
    control_sent = (pi_to_pixhawk_clock(fd.timing_data[i][5]) - px4_start) / 1000

    print(state_sample, state_sent, state_receive, nmpc_start, nmpc_end, control_sent)

    ax.broken_barh([(state_sample, state_sent - state_sample)], (i-0.4,0.8), facecolors='tab:blue')
    ax.broken_barh([(state_sent, state_receive - state_sent)], (i-0.4,0.8), facecolors='tab:red')
    ax.broken_barh([(state_receive, nmpc_start - state_receive)], (i-0.4,0.8), facecolors='tab:orange')
    ax.broken_barh([(nmpc_start, nmpc_end - nmpc_start)], (i-0.4,0.8), facecolors='tab:green')
    ax.broken_barh([(nmpc_end, control_sent - nmpc_end)], (i-0.4,0.8), facecolors='tab:purple')


ax.set_xlabel("Time (ms)")
ax.set_ylabel("Control Cycle")
ax.set_title("Timing of 20 Control Cycles")
ax.set_yticks(range(n_cycles))
ax.grid(axis="x")

plt.show()