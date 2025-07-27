#!/bin/bash

scp izzy@192.168.0.100:~/drone_ws/plotter_logs/current.json ./
python hop/plotter.py