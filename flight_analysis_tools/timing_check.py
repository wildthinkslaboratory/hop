import matplotlib.pyplot as plt
import numpy as np
from flight_analysis_tools.flight_data import FlightData


fd = FlightData()
px4_start = fd.timing_data[0][0]


fig, ax = plt.subplots(figsize=(12,6))

# np.random.seed(0)


n_cycles = len(fd.timing_data)-1
n_cycles = 50


# we map our first pi time to the time of the first pixhawk send time 
# plus a time delta for the message time.
def pi_to_pixhawk_clock(pi_time):
    px4_pi_delta = 5000 # 
    px4_first_sample_time = fd.timing_data[0][0]
    first_pi_time = fd.timing_data[0][1]
    
    # pi time is in 10^-6 seconds and px4 is in 10^-5 seconds
    return px4_first_sample_time + px4_pi_delta + (pi_time - first_pi_time)
    


for i in range(n_cycles):
    # print(fd.timing_data[i])
    # first get everything into ms
    sample = (fd.timing_data[i][0] - px4_start) / 1000
    main_receive = (pi_to_pixhawk_clock(fd.timing_data[i][1]) - px4_start) / 1000
    main_sent = (pi_to_pixhawk_clock(fd.timing_data[i][2]) - px4_start) / 1000
    nmpc_receive = (pi_to_pixhawk_clock(fd.timing_data[i][3]) - px4_start) / 1000
    nmpc_time = fd.timing_data[i][4] * 1000


    # print(sample, main_receive, main_sent, nmpc_time)

    ax.broken_barh([(sample, main_receive - sample)], (i-0.4,0.8), facecolors='tab:blue')
    ax.broken_barh([(main_receive, main_sent - main_receive)], (i-0.4,0.8), facecolors='tab:red')
    ax.broken_barh([(main_sent, nmpc_receive - main_sent)], (i-0.4,0.8), facecolors='tab:orange')
    ax.broken_barh([(nmpc_receive, nmpc_time)], (i-0.4,0.8), facecolors='tab:green')
    ax.broken_barh([(nmpc_receive + nmpc_time, 30)], (i-0.4,0.8), facecolors='tab:purple')
    # ax.broken_barh([(control_sent, 30)], (i-0.4,0.8), facecolors='tab:pink')


ax.set_xlabel("Time (ms)")
ax.set_ylabel("Control Cycle")
ax.set_title("Timing of 20 Control Cycles")
ax.set_yticks(range(n_cycles))
ax.grid(axis="x")

plt.show()