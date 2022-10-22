import numpy as np
import os
from MDP.GridWorld import GridWorld

#Grid block code
# 0 -> regular state with default reward -0.04
# 1 -> +ve exist state with default reward +1.0
# 2 -> -ve exist state with default reward -1.0
# 3 -> blocking state which is not accessible by default

SELECT_GRID = '3x4'
GRIDWORLD_PLOTS_DIR = 'plots/GridWorld'
if not os.path.exists(GRIDWORLD_PLOTS_DIR):
    os.makedirs(GRIDWORLD_PLOTS_DIR)

if SELECT_GRID == '3x4':
    REWARD_VALUES = {0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}
    problem = GridWorld('data/world00.csv', reward=REWARD_VALUES, random_rate=0.2)
    problem.plot_map(fig_size=(10, 8), 
    SHOW_STATES=True, 
    SHOW_REWARDS=True,
    SAVE_PLOTS=True,
    fil_name=os.path.join(GRIDWORLD_PLOTS_DIR, 'rewards.png'))
else:
    REWARD_VALUES = {0: -0.04, 1: 10.0, 2: -10.0, 3: np.NaN}
    problem = GridWorld('data/world01.csv', reward=REWARD_VALUES, random_rate=0.2)
    problem.plot_map(fig_size=(10, 8))

problem.plot_transition_probabilities(fig_size=(20, 10),
                    SAVE_PLOTS=True, fil_name=os.path.join(GRIDWORLD_PLOTS_DIR, 'transition_prob.png'))

init_policy = problem.generate_random_policy()
problem.plot_policy(init_policy, fig_size=(20, 10),
                    SAVE_PLOTS=True, fil_name=os.path.join(GRIDWORLD_PLOTS_DIR, 'policy.png'))
problem.plot_values(init_policy, np.zeros(problem.num_states), fig_size=(10, 8),
                                SAVE_PLOTS=True, fil_name=os.path.join(GRIDWORLD_PLOTS_DIR, 'values.png'))

reward_function = problem.reward_function
print(f'reward function =')
for s in range(len(reward_function)):
    print(f'State s = {s}, Reward R({s}) = {reward_function[s]}')

transition_model = problem.transition_model
print(f'transition model =')
for s in range(transition_model.shape[0]):
    print('======================================')
    for a in range(transition_model.shape[1]):
        print('--------------------------------------')
        for s_prime in range(transition_model.shape[2]):
            print(f's = {s}, a = {a}, s\' = {s_prime}, p = {transition_model[s, a, s_prime]}')
