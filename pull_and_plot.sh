git pu#!/bin/bash

scp izzy@192.168.0.100:~/drone_ws/plotter_lots/current.json ./plotter_logs/current.json
python plotter.py