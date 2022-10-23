import numpy as np
import os
from copy import deepcopy
import sys
sys.path.append('..')

from MDP.GridWorld import GridWorld
from MDP.ValueIteration import ValueIteration

problem = GridWorld('../data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}, random_rate=0.2)

VI_PLOTS_DIR = 'plots/ValueIteration'
if not os.path.exists(VI_PLOTS_DIR):
    os.makedirs(VI_PLOTS_DIR)

l_gamma = [0.1, 0.5, 0.9]
for gamma in l_gamma:
    solver = ValueIteration(problem, gamma=gamma)
    
    _save_plots = True if gamma == 0.9 else False
    solver.train(plot_save_freq=15, save_plots=_save_plots, save_dir=VI_PLOTS_DIR)
    
    problem.plot_values(policy=solver.policy, values=solver.values, SAVE_PLOTS=True,
                        fil_name=os.path.join(VI_PLOTS_DIR, f"vi_values_gamma{gamma}.png"))

for i, _policy in enumerate(solver.policy_history):
    fil_name=os.path.join(VI_PLOTS_DIR, f"rewards_histograms{str(i).zfill(3)}.png")
    problem.random_start_policy(policy=_policy, start_pos=(2, 0), n=1000,
                                SAVE_PLOTS=True, fil_name=fil_name)
