import numpy as np
from PolicyIteration import PolicyIteration


class ADPLearner:
    def __init__(self, problem, gamma=0.9, epsilon=0.9, xi=0.99):
        self.problem = problem
        num_states = problem.num_states
        num_actions = problem.num_actions
        self.num_states = problem.num_states
        self.num_actions = problem.num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi = xi

        self.u_table = np.zeros(num_states)
        self.r_table = np.zeros(num_states)
        self.p_table = np.zeros((num_states, num_actions, num_states))
        self.cur_policy = np.random.randint(num_actions, size=num_states)
        self.visited_state = np.zeros(num_states)
        self.count_action = np.zeros((num_states, num_actions))
        self.count_outcome = np.zeros((num_states, num_actions, num_states))

    def percept(self, s, a, s_prime, r):
        if self.visited_state[s_prime] == 0:
            self.u_table[s_prime] = r
            self.r_table[s_prime] = r
            self.visited_state[s_prime] = 1
        self.count_action[s, a] += 1
        self.count_outcome[s, a, s_prime] += 1
        self.p_table[s, a] = self.count_outcome[s, a] / self.count_action[s, a]

    def actuate(self, s_prime):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return self.cur_policy[s_prime]

    def policy_update(self):
        solver = PolicyIteration(self.problem, self.gamma,
                                 init_policy=self.cur_policy, init_value=self.u_table)
        solver.train(tol=1e-3, save_plots=False)
        self.cur_policy = solver.policy
        self.epsilon *= self.xi












