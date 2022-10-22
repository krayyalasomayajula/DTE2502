import numpy as np
import os
import itertools
from copy import deepcopy

from MDP.GridWorld import GridWorld
from MDP.PolicyIteration import PolicyIteration

problem = GridWorld('data/world00.csv', reward={0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}, random_rate=0.2)
policy = [1, 1, 3, 1, 0, 0, 2, 0, 1, 2, 1, 0]

PI_PLOTS_DIR = 'plots/PolicyIteration'
if not os.path.exists(PI_PLOTS_DIR):
    os.makedirs(PI_PLOTS_DIR)

solver = PolicyIteration(problem, gamma=0.9, init_policy=policy)
solver.train(plot_save_freq=1, save_dir=PI_PLOTS_DIR)

problem.plot_values(policy=solver.policy, values=solver.values, SAVE_PLOTS=True)
    
for i, _policy in enumerate(solver.policy_history):
    fil_name=os.path.join(PI_PLOTS_DIR, f"rewards_histograms{str(i).zfill(3)}.png")
    problem.random_start_policy(policy=_policy, start_pos=(2, 0), n=1000,
                                SAVE_PLOTS=True, fil_name=fil_name)
