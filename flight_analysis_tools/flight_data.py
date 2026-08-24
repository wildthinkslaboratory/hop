from hop.utilities import import_data
import sys
import numpy as np
from plotting.plots import plot_state, plot_control, plot_pwm, plot_attitude, plot_parameters
from hop.utilities import quaternion_to_angle

import matplotlib.pyplot as plt

class FlightData:
    def __init__(self, filename='./plotter_logs/current.json'):
        
        # read in logfile and time point to begin analyzing
        self.dt = 0.02
        self.start_time = 0.0
        self.log_file_name = filename
        if len(sys.argv) > 1:
            self.log_file_name = sys.argv[1]
            self.start_time = float(sys.argv[2])
            print(self.log_file_name, self.start_time)


        log = import_data(self.log_file_name)   
        self.constants = log['constants']
        data = log['run_data']

        # read in the flight data
        self.state_data = np.empty([len(data),13])
        self.control_data = np.empty([len(data),4])
        self.pwm_motors = np.empty([len(data),2])
        self.pwm_servos = np.empty([len(data),2])
        self.parameters = np.empty([len(data),5])
        self.attitude = np.empty([len(data),3])
        self.timing_data = []
        self.voltage = []
        self.timestamps = []

        # collect all the data into arrays
        for i, d in enumerate(data):
            self.state_data[i] = np.array(d['state'])
            self.control_data[i] = np.array(d['control'])
            self.pwm_motors[i] = np.array(d['pwm_motors'])
            self.pwm_servos[i] = np.array(d['pwm_servos'])
            if 'timing' in d:
                self.timing_data.append(d['timing'])
            if len(d['parameters']) == 4:
                self.parameters[i] = np.array(d['parameters'] + [0.0])
            else:
                self.parameters[i] = np.array(d['parameters'])
            self.voltage.append(d['parameters'][3])

            # turn quaternions into attitude
            q = np.reshape(self.state_data[i][6:10].copy(), (4,))
            self.attitude[i] = quaternion_to_angle(q)


        stop_index = int(self.start_time // self.dt)
        self.len_used_data = len(data) - stop_index -1

        # Truncate the data to start at the takeoff
        self.state_data = self.state_data[stop_index+1:]
        self.control_data = self.control_data[stop_index+1:]
        self.voltage = self.voltage[stop_index+1:]
        self.timestamps = self.timestamps[stop_index+1:]
        self.pwm_motors = self.pwm_motors[stop_index+1:]
        self.pwm_servos = self.pwm_servos[stop_index+1:]
        self.parameters = self.parameters[stop_index+1:]
        self.timing_data = self.timing_data[stop_index+1:]


    def plot(self, begin=0, end=-1):

        if end == -1:
            end = self.len_used_data

        # verify our ranges
        assert(begin >= 0 and begin < self.len_used_data)
        assert(end >= 0 and end <= self.len_used_data)

        tspan = np.arange(0, (end-begin) * self.dt , self.dt)
        plot_state(tspan, self.state_data[begin:end], 'flight data state')
        plot_control(tspan, self.control_data[begin:end], 'flight data control')
        plot_parameters(tspan, self.parameters[begin:end], 'flight parameters')
        plot_pwm(tspan, self.pwm_servos[begin:end], self.pwm_motors[begin:end], 'flight data pwm')
        plot_attitude(tspan, self.attitude[begin:end], 'flight data attitude')
        plt.show()