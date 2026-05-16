from hop.utilities import import_data

log = import_data('./plotter_logs/current.json')
data = log['run_data']

state_data = [d['state'] for d in data]

import csv

filename = 'run_data.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(state_data)