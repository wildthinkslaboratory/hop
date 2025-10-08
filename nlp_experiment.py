#
# This runs drone simulations, plots results and gives timing summaries
#
from hop.drone_model import DroneModel
from hop.dompc import DroneMPC
from hop.constants import Constants
from do_mpc.simulator import Simulator
import casadi as ca
from do_mpc.estimator import StateFeedback
import numpy as np
import statistics as stats
from hop.utilities import import_data
from time import perf_counter
from hop.multiShooting import DroneNMPCMultiShoot
from hop.chebyshev_ps import DroneNMPCwithCPS
from animation import RocketAnimation
import matplotlib.pyplot as plt
from plots import plot_time_comparison, plot_state_for_paper, plot_control_for_paper

mc = Constants()


# If you just want to run a single test you can loop over this list
single_test = [
  {
    "x0": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-1, -0.1, -0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 250,
    "title": "y115dx"
  },
]


# Here is the full set of tests if you want to run all the simulations
test_list_for_paper = [
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [0.0, -0.2, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.8,
    "num_iterations": 200,
    "title": "hover"
  },
  {
    "x0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.383, 0.924, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-0.2, -0.5, 0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.8,
    "num_iterations": 500,
    "title": "45dz"
  },
  {
    "x0": [1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [1, -0.5, -1],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 200,
    "title": "x1z1vx"
  },
  {
    "x0": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.259, 0.0, 0.0, 0.966, 0.0, 0.0, 0.0],
    "xr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "animation_forward": [-1, -0.1, -0.2],
    "animation_up": [0, 1, 0],
    "animation_frame_rate": 0.4,
    "num_iterations": 250,
    "title": "y115dx"
  },
]


# which nlp formulations to run
# oc - orthogonal collocation by dompc
# cps - Chebyshev pseudospectral collocation
# ms - multiple shooter with Runge-Kutta
nlps_to_run = ['oc', 'cps', 'ms']

for test in test_list_for_paper:

    # set up the test case
    num_iterations = test['num_iterations']
    x_init = ca.DM(test['x0'])
    xr = ca.DM(test['xr'])
    tspan = np.arange(0,num_iterations* mc.dt,mc.dt)

    # data structures to collect experimental data in
    state_data = {}
    control_data = {}
    time_data = {}
    for nlp in nlps_to_run:
      state_data[nlp] = np.empty([num_iterations,13])
      control_data[nlp] = np.empty([num_iterations,4])
      time_data[nlp] = []

    if 'oc' in nlps_to_run:
      # first we set up the do-mpc solver
      # it uses orthagonal collocation
      model = DroneModel()
      mpc = DroneMPC(mc.dt, model.model)
      estimator = StateFeedback(model.model)
      sim = Simulator(model.model)
      sim.set_param(t_step = mc.dt)

      # this is annoying but necessary
      # the model has a parameter for waypoints
      # it's used in the mpc cost function only
      # the simulator get's confused if it has undefined parameters
      # so we give it a dummy function
      p_template = sim.get_p_template()
      def dummy(t_now):
          p_template['p_goal'] = np.array([0.0, 0.0, 0.0])
          return p_template
      sim.set_p_fun(dummy)

      sim.setup()
      mpc.setup_cost()
      mpc.set_start_state(x_init)
      sim.x0 = x_init
      estimator.x0 = x_init
      x0 = x_init

      # run do-mpc solver
      print('running do-mpc solver')
      for k in range(num_iterations):
          start_time = perf_counter()
          # mpc.set_waypoint(np.array([0.0, 0.0, 0.0]))
          u0 = mpc.mpc.make_step(x0)
          step_time = perf_counter() - start_time

          
          y_next = sim.make_step(u0)
          x0 = estimator.make_step(y_next)

          state_data['oc'][k] = np.reshape(x0, (13,))
          control_data['oc'][k] = np.reshape(u0, (4,))
          time_data['oc'].append(step_time)
          

    if 'cps' in nlps_to_run:
      # set up the Chebyshev pseudospectral nmpc solver
      cheb_nmpc = DroneNMPCwithCPS()
      cheb_nmpc.set_goal_state(xr)
      cheb_nmpc.set_start_state(x_init)
      x0 = x_init
      u0 = np.zeros(4)


      print('running Chebyshev pseudospectral nmpc solver')
      for k in range(num_iterations):

          start_time = perf_counter()
          # Solve the NMPC for the current state x_current
          u0 = cheb_nmpc.make_step(x0, u0, np.array([0.0, 0.0, 0.0]))
          step_time = perf_counter() - start_time

          # Propagate the system using the discrete dynamics f (Euler forward integration)
          x0 = x0 + mc.dt* cheb_nmpc.f(x0,u0)
          
          state_data['cps'][k] = np.reshape(x0, (13,))
          control_data['cps'][k] = np.reshape(u0, (4,))
          time_data['cps'].append(step_time)

    if 'ms' in nlps_to_run:
      # run the multiple shooter nmpc
      ms_nmpc = DroneNMPCMultiShoot()
      ms_nmpc.set_goal_state(xr)
      ms_nmpc.set_start_state(x_init)
      x0 = x_init

      print('running multiple shooter nmpc solver')
      for k in range(num_iterations):

          start_time = perf_counter()
          # Solve the NMPC for the current state x_current
          u0 = ms_nmpc.make_step(x0, u0, np.array([0.0, 0.0, 0.0]))
          step_time = perf_counter() - start_time

          # Propagate the system using the discrete dynamics f (Euler forward integration)
          x0 = x0 + mc.dt* ms_nmpc.f(x0,u0)
          
          state_data['ms'][k] = np.reshape(x0, (13,))
          control_data['ms'][k] = np.reshape(u0, (4,))
          time_data['ms'].append(step_time)


    # compute statistics for the timing of the nmpc calls
    mean_time = [round(stats.mean(time_data[nlp]),3) for nlp in nlps_to_run]
    max_time = [round(max(time_data[nlp]),3) for nlp in nlps_to_run]

    # print timing results
    print(test['title'])
    print("          {: >20} {: >20} {: >20}".format(*nlps_to_run))
    print("-----------------------------------------------------------------------------------------")
    print("mean      {: >20} {: >20} {: >20}".format(*mean_time))
    print("max       {: >20} {: >20} {: >20}".format(*max_time))
 

    for i, nlp in enumerate(nlps_to_run):
      plot_state_for_paper(tspan, state_data[nlp], test["title"], i+1)
  
    start_id = len(nlps_to_run)+1
    for i, nlp in enumerate(nlps_to_run):
      plot_control_for_paper(tspan, control_data[nlp], test["title"], i+start_id)

    plot_time_comparison(tspan, time_data, test["title"], 7)
    plt.show()

