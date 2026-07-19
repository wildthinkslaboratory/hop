from hop.utilities import import_data
import sys
import numpy as np


class FlightData:
    def __init__(self):
        
        # read in logfile and time point to begin analyzing
        self.dt = 0.02
        self.start_time = 0.0
        self.log_file_name = './plotter_logs/current.json'
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
        self.timing_data = []
        self.voltage = []
        self.timestamps = []

        # collect all the data into arrays
        for i, d in enumerate(data):
            self.state_data[i] = np.array(d['state'])
            self.control_data[i] = np.array(d['control'])
            self.voltage.append(d['voltage'])
            self.timestamps.append(d['timestamp'])
            self.pwm_motors[i] = np.array(d['pwm_motors'])
            self.pwm_servos[i] = np.array(d['pwm_servos'])
            if 'timing' in d:
                self.timing_data.append(d['timing'])
            if len(d['parameters']) == 4:
                self.parameters[i] = np.array(d['parameters'] + [0.0])
            else:
                self.parameters[i] = np.array(d['parameters'])

        stop_index = int(self.start_time // self.dt)
        self.len_used_data = len(data) - stop_index -1

        # Truncate the data to start at the takeoff
        self.state_data = self.state_data[stop_index+1:]
        self.control_data = self.control_data[stop_index+1:]
        self.voltage = self.voltage[stop_index+1:]
        self.timestamps = self.timestamps[stop_index+1:]
        self.pwm_motors = self.pwm_motors[stop_index+1:]
        self.pwm_servos = self.pwm_servos[stop_index+1:]
        self.parameters = self.parameters[stop_index+1:-1]
        self.timing_data = self.timing_data[stop_index+1:]