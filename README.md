# TVC Drone Project Code

## Latest Updates

### Fixed the time delay

We did some more formal testing of the servos to identify the time delay. We set up an LED on the Pi to signal the time the servo PWM signal is sent on the Pi, and then did slow motion videos to observe the servo response. The JX PDI-6221MG-120 is really slow and not terribly precise. We ordered the KST BLS815 V8 servos and reran the tests and found them to be much faster and more precise.

We ran a ramp test. We sent a series of servo commands ramping from 0 degrees up to 20 degrees then back down to -20 degrees and up to 0 degrees. This was repeated for a series of cycles. The commands were sent at 50Hz with each command being for a 4 degree change. The Pi turns the LED on when it begins sending the upward trajectory and turns the LED off when it starts sending the downward commands.

| Servo             | time to movement | ramp lag | ramp degree |
| ----------------- | ---------------- | -------- | ----------- |
| JX PDI-6221MG-120 | 33 ms            | 80 ms    | 10          |
| KST BLS815 V8     | 30 ms            | 33 ms    | 20          |

Both servos had about the same amount of time between when the first message was sent and when the gimbal started moving. This was about 30 ms. As the ramp continued, the JX PDI-6221MG-120 developed a constant 80 ms lag and failed to reach the required 20 degrees before changing direction. It stopped at around 10 degrees. The KST BLS815 V8 developed a constant lag of 33 ms and always reached the full 20 degrees of rotation before changing direction.

[First Successful Test Flight](https://www.youtube.com/watch?v=N2oum2yvaio)

[Close Up of Gimbal Action](https://www.youtube.com/watch?v=m86OpVHrvyQ)

You can see from our test flight that although the drone balances, it oscillates around a balance point. We don't have the precise control that we need. We were able to identify a time delay in the servos by looking at the flight data and comparing it with the expected behavior based on the equations of motion. We've confirmed that the delay comes mostly from actuation of the servos. We had to make a slow motion video of the gimbals responding to commands from the Pi with an LED that lights when the first message is sent. There's about a 30ms delay between when the Pi sends the message and when the servos start to move. The servos then get further behind leading to about a 120 ms delay. The servos we used were JX PDI-6221MG-120 which are used primarily in small remote control cars. We've bought a couple of KST BLS815 V8 servos which are much better so we'll see what kind of delay we see with those. Then we'll model what ever hopefully smaller delay we have in our equations of motion.

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
