# TVC Drone Project Code

## Project Structure

- `documents` directory contains latex files for papers
- `experiments` directory contains code to run experiments for paper
- `flight_analysis_tools` directory contains code for analyzing flight data for the drone
- `hop` directory contains code for running NMPC control algorithm on drone with ros2, three different NLP formulations for the NMPC, and code to test the drone servos and motors.
- `plots` directory holds plot pdfs produced from various programs
- `plotter_logs` contains flight data downloaded from the Raspberry Pi
- `resource` directory is used by ros2 build
- `simulation_tools` directory contains code for running simulations and testing the NMPC algorithm
- `tools` directory contains general things that don't fit anywhere else

## Setup for Running Simulations, Experiments and Flight Analysis

In the top level `hop` directory build a virtual python environment and activate it

```
python3 -m venv venv
source venv/bin/activate
```

Now we can install needed libraries

```
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install matplotlib
python -m pip install casadi
python -m pip install 'do-mpc[full]'

```

For the animation you need

```
python -m pip install vpython
python -m pip install numpy-quaternion
```

## Running Experiments

Open file `run_experiments.py` and uncomment the experiment you want to run. It is not recommended to run them all at once. Run the experiment in the top level directory by running

```
python run_experiments.py
```

## Running Simulations

## Running Flight Analysis

You can also get flight data from the PixHawk by connecting to QGroundControl.

## Running the NMPC Controller on the Pi

### Start the MicroAgent

Open up a shell on the Pi with `ssh`.

```
ssh izzy@X.X.X.X
```

Run the MicroAgent

```
sudo /usr/local/bin/MicroXRCEAgent serial --dev /dev/ttyAMA0 -b 921600
```

When the px4 is connected you should see topics coming in.

### VSCode Window

`ssh` into the Pi through vscode so you can edit and use git.

### Run the controller

Open up another shell on the Pi with `ssh`.

```
cd drone_ws
colcon build
source install/setup.bash
ros2 run hop nmpc_controller
```

You can run the nodes `test_motors` and `test_servos` this way as well.

**Recommendation:** Don't do anything other than run the code from this shell. If you want to do `git` stuff, do it from a different shell login. I've really screwed up our directories because I dropped down into the `src/hop` folder to do stuff, then forgot what directory I'm in and ran `colcon build`. It generates files where they shouldn't be and they end up in the git directory. We added them to the git repo by accident and it was a pain to sort it all out.

### Keyboard Controls for Flights

All of these default to `logging = False` (meaning don't log messages to the console) so you can feed in waypoints or pwm from the keyboard without seeing streams of log messages.

- `nmpc_controller`: Use the `u` key to progress forward through waypoints.
- `test_motors`: Use the `u` key to increment pwm by $0.1$.
- `test_servos`: Use the `u` key to increment pwm by $0.1$ and `j` to decrement by $0.1$

**Hit any other keyboard key to exit the run**

### Logging

Flight data is automatically logged and logs are written to `plotter_logs` folder. Logs are saved to file and named with the datetime they were created and the most current log is also stored in `current.json`. You can pull them to your mac with the `pull_and_plot.sh` bash shell and run the flight analysis tools on them. It has an old IP address for the Pi so you can edit it or just copy the command.

Plot the most current run with

```
python plot_run.py
```

Make sure to source your `hop` directory on your mac first.

```
source venv/bin/activate
```
