#!/bin/bash


if [ ! -d "plotter_logs" ]; then
  echo "directory plotter_logs does not exist. Creating it."
  mkdir plotter_logs
fi

scp izzy@192.168.0.100:~/drone_ws/src/hop/plotter_logs/current.json ./plotter_logs/
