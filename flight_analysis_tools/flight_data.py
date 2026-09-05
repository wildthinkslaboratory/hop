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
        self.end_time = None
        if len(sys.argv) > 1:
            self.log_file_name = sys.argv[1]
            self.start_time = float(sys.argv[2])
            print(self.log_file_name, self.start_time)

        if len(sys.argv) > 3:
            self.end_time = float(sys.argv[3])
            

        log = import_data(self.log_file_name)   
        self.constants = log['constants']
        data = log['run_data']

        # read in the flight data
        if len(data[0]['raw_state']) == 13:
            self.state_data = np.empty([len(data),13])
            self.future_state_data = np.empty([len(data),13])
        else:
            self.state_data = np.empty([len(data),14])
            self.future_state_data = np.empty([len(data),14])

        self.control_data = np.empty([len(data),4])
        self.pwm_motors = np.empty([len(data),2])
        self.pwm_servos = np.empty([len(data),2])
        self.parameters = np.empty([len(data),5])
        self.attitude = np.empty([len(data),3])
        self.timing_data = []
        self.raw_voltage = np.zeros([len(data),1])
        self.current = np.zeros([len(data),1])
        self.timestamps = []

        # collect all the data into arrays
        for i, d in enumerate(data):
            self.state_data[i] = np.array(d['raw_state'])
            self.future_state_data[i] = np.array(d['state'])
            self.control_data[i] = np.array(d['control'])
            self.pwm_motors[i] = np.array(d['pwm_motors'])
            self.pwm_servos[i] = np.array(d['pwm_servos'])
            if 'timing' in d:
                self.timing_data.append(d['timing'])
            if len(d['parameters']) == 4:
                self.parameters[i] = np.array(d['parameters'] + [0.0])
            else:
                self.parameters[i] = np.array(d['parameters'])

            if 'raw_voltage' in d:
                self.raw_voltage[i] = d['raw_voltage']
                self.current[i] = d['current_a']
      
            
            # turn quaternions into attitude
            q = np.reshape(self.state_data[i][6:10].copy(), (4,))
            self.attitude[i] = quaternion_to_angle(q)


        stop_index = int(self.start_time // self.dt)
        end_index = len(data)
        if not self.end_time == None: 
            end_index = int(self.end_time // self.dt)

        self.len_used_data = end_index - stop_index -1

        # Truncate the data to start at the takeoff
        self.state_data = self.state_data[stop_index+1: end_index]
        self.future_state_data = self.future_state_data[stop_index+1:end_index]
        self.control_data = self.control_data[stop_index+1:end_index]
        self.pwm_motors = self.pwm_motors[stop_index+1:end_index]
        self.pwm_servos = self.pwm_servos[stop_index+1:end_index]
        self.parameters = self.parameters[stop_index+1:end_index]
        self.attitude = self.attitude[stop_index+1:end_index]
        self.timing_data = self.timing_data[stop_index+1:end_index]
        self.current = self.current[stop_index+1:end_index]
        self.raw_voltage = self.raw_voltage[stop_index+1:end_index]
        self.timestamps = self.timestamps[stop_index+1:end_index]



    def plot(self, begin=0, end=-1, plots=[], i=1):

        if end == -1:
            end = self.len_used_data

        # verify our ranges
        assert(begin >= 0 and begin < self.len_used_data)
        assert(end >= 0 and end <= self.len_used_data)

        tspan = np.arange(0, (end-begin) * self.dt , self.dt)
        if not len(tspan) == end-begin:
            tspan = tspan[:-1]
        if 'raw_voltage' in plots or plots == []:
            plt.figure(i)
            i += 1
            plt.plot(tspan, self.raw_voltage[begin:end])
            plt.plot(tspan, self.parameters[begin:end, 3])
            plt.title('raw voltage')

        if 'current_a' in plots or plots == []:    
            plt.figure(i)
            i += 1
            plt.plot(tspan, self.current[begin:end])
            plt.title('raw current')

        if len(self.state_data[0]) == 14:
            plt.figure(i)
            i += 1
            plt.plot(tspan, self.state_data[begin:end, 13])
            plt.title('predicted thrust')

        if 'state' in plots or plots == []: 
            plot_state(tspan, self.state_data[begin:end], 'flight data state')
        if 'future_state' in plots or plots == []: 
            plot_state(tspan, self.future_state_data[begin:end], 'flight data future state')
        if 'control' in plots or plots == []: 
            plot_control(tspan, self.control_data[begin:end], 'flight data control')
        if 'parameters' in plots or plots == []: 
            plot_parameters(tspan, self.parameters[begin:end], 'flight parameters')
        if 'pwm' in plots or plots == []: 
            plot_pwm(tspan, self.pwm_servos[begin:end], self.pwm_motors[begin:end], 'flight data pwm')
        if 'attitude' in plots or plots == []: 
            plot_attitude(tspan, self.attitude[begin:end], 'flight data attitude')
