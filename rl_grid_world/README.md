This repo aims to introduce reinforcement learning through "A grid world problem from the book *Artificial Intelligence A Modern Approach* by Stuart Russell and Peter Norvig".

This implemented is based on: https://github.com/clumsyhandyman/mad-from-scratch/tree/main/MarkovDecissionProcess
The code changes are made to make the analysis more clear and pedagogical.

# MDP-with-grid-world

The following algorithms are implemented in Python:
- Markov Decission Precoss (MDP) and Reinforcement Learning (RL)
	- Environment/World is known: (run tests from `known_world` directory)
		- Basics of MDP (grid world)
		- Policy Iteration
		- Value Iteration
	- Environment/World is unknown: Learning through a model approximation of it (run tests from `unknown_world_model_based` directory)
		- Model-based Adaptive Dynamic Programming (ADP)
	- Environment/World is unknown: Learning through a model free approach (run tests from `unknown_world_model_free` directory)
		- Model-free Monte Carlo (MC)
	- Environment/World is unknown: Temporal difference learning (run tests from `unknown_world_tdl` directory)
		- Q-Learning

### GridWorld.py

Under the class **GridWorld**, the following functions are provided:
- *get_state_from_pos(pos)*: transfer a position in the grid world into an integer representing state.
- *get_pos_from_state(state)*: transfer an integer representing state back into a position in the grid world.
- *get_reward_function()*: calculate the reward function r(s) of MDP.
- *get_transition_model()*: calculate the transitional model p(s'|s, a) of MDP.
- *generate_random_policy()*: initialize a policy of random actions.
- *execute_policy(policy, start_pos)*: get the total reward starting from the start_pos following the given policy.
- *random_start_policy(policy, n)*: repeatedly execute the given policy for n times.
- *blackbox_move(s, a)*: simulate an environment where the agent can not access the reward function and transition model. The agent provides the current state s and an action a, this function returns the next state s' of the agent and the reward assigned to the agent through this move.
- *plot_map()*: visualize the map of the grid world with states and rewards.
- *plot_policy(policy)*: visualize the given policy.
- *plot_values(policy, values)*: visualize the given policy and utility values
- *plot_transition_probabilities()*: visualize the map state-actions with transistion probabilities.

### PolicyIteration.py
Implement policy iteration to solve a MDP.

Under the class **PolicyIteration**, the following functions are provided:
- *one_policy_evaluation()*: perform one sweep of policy evaluation.
- *run_policy_evaluation(tol)*: perform sweeps of policy evaluation iteratively with a stop criterion of the given tol.
- *run_policy_improvement()*: perform one policy improvement.
- *train(tol, plot)*: perform policy iteration by iteratively alternates policy evaluation and policy improvement. If plot is true, the function plots learning curves showing number of sweeps in each iteration and number of policy updates in each iteration.

### ValueIteration.py
Implement value iteration to solve a MDP.

Under the class **ValueIteration**, the following functions are provided:
- *one_iteration()*: perform one iteration of value evaluation.
- *get_policy()*: determine policy based on current utility.
- *train(tol, plot)*: perform value iteration with a stop criterion of the given tol. If plot is true, the function plots learning curves showing maximum value change in each iteration.

