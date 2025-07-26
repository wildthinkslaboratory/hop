#!/bin/bash

scp izzy@192.168.0.100:~/drone_ws/plotter_lots/log.json ./plotter_logs/log.json
python plotter.py