### Running the NMPC Controller on the Pi

#### Start the MicroAgent

Open up a shell on the Pi with `ssh`.

```
ssh izzy@X.X.X.X
```

Run the MicroAgent

```
sudo /usr/local/bin/MicroAgent serial --dev /dev/ttyAMA0 -b 921600
```

When the px4 is connected you should see topics coming in.

#### VSCode Window

`ssh` into the Pi through vscode so you can edit and use git.

#### Run the controller

Open up another shell on the Pi with `ssh`.

```
cd drone_ws
colcon build
source install/setup.bash
ros2 run hop nmpc_controller
```

You can run the nodes `test_motors` and `test_servos` this way as well.

#### Final Notes

All of these default to `logging = False` so you can feed in waypoints or pwm from the keyboard.

- `nmpc_controller`: Use the `u` key to progress forward through waypoints.
- `test_motors`: Use the `u` key to increment pwm by $0.1$.
- `test_servos`: Use the `u` key to increment pwm by $0.1$ and `j` to decrement by $0.1$

**Hit any other keyboard key to exit the run**

### Logging

Logs are written to `plotter_logs` folder. Logs are saved to file and named with the datetime they were created and the most current log is also stored in `current.json`. You can pull them to your mac with the `pull_and_plot.sh` bash shell. It has an old IP address for the Pi so you can edit it or just copy the command.

Plot the most current run with

```
python plot_run.py
```

Make sure to source your `hop` directory on your mac first.

```
source venv/bin/activate
```
