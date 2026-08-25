from flight_analysis_tools.flight_data import FlightData
import matplotlib.pyplot as plt

fd = FlightData()




# import numpy as np
# from collections import deque
# import matplotlib.pyplot as plt

# len_data = len(fd.raw_voltage)
# roll_voltage = np.zeros([len_data,1])
# median_filter = np.zeros([len_data,1])
# low_pass = np.zeros([len_data,1])
# voltage_history = deque(maxlen=8)
# median_history = deque(maxlen=8)
# alpha = 0.92

# for i in range(len_data):
#     median_history.append(fd.raw_voltage[i])
#     voltage_history.append(fd.raw_voltage[i])
#     roll_voltage[i] = np.mean(voltage_history)
#     median_filter[i] = np.median(median_history)

#     if i == 0:
#         low_pass[i] = fd.raw_voltage[i]
#     else:
#         low_pass[i] = alpha * low_pass[i-1] + (1 - alpha) * fd.raw_voltage[i]

# tspan = np.arange(0, len_data * fd.dt , fd.dt)
# print(len(fd.raw_voltage), len(fd.parameters))
# plt.figure(1)
# plt.plot(tspan, fd.raw_voltage)
# plt.plot(tspan, fd.parameters[:,3])
# # plt.plot(tspan, roll_voltage)
# plt.plot(tspan, low_pass)

# plt.title('voltage')

fd.plot()
plt.show()