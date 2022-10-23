import numpy as np
import matplotlib.pyplot as plt
from os import path

class ValueIteration:
    def __init__(self, problem, gamma):
        self.problem = problem
        transition_model = problem.transition_model
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.reward_function = np.nan_to_num(problem.reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
        self.values = np.zeros(self.num_states)
        self.policy = None
        self.policy_history = []

    def one_iteration(self):
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

            self.values[s] = max(v_list)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def get_policy(self):
        pi = np.ones(self.num_states) * -1
        for s in range(self.num_states):
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)

            max_index = []
            max_val = np.max(v_list)
            for a in range(self.num_actions):
                if v_list[a] == max_val:
                    max_index.append(a)
            pi[s] = np.random.choice(max_index)
        return pi.astype(int)

    def train(self, tol=1e-3, save_plots=True, 
                plot_save_freq=20, 
                save_dir=''):
        epoch = 0
        delta = self.one_iteration()
        delta_history = [delta]

        if save_plots:
            policy = self.get_policy()
            self.policy_history = [policy]

        while delta > tol:
            epoch += 1
            delta = self.one_iteration()
            delta_history.append(delta)
            if epoch % plot_save_freq == 0:
                policy = self.get_policy()
                self.policy_history.append(policy)

                if save_plots:
                    fil_name = path.join(save_dir, f"policy_epoch{str(epoch).zfill(3)}.png")
                    self.problem.plot_policy(policy, fig_size=(20, 10),
                        SAVE_PLOTS=True, fil_name=fil_name)

            if delta < tol:
                break
        self.policy = self.get_policy()

        if not epoch % plot_save_freq == 0:
            self.policy_history.append(self.policy)
        if save_plots:
            fil_name = path.join(save_dir, f"policy_epoch_final.png")
            self.problem.plot_policy(self.policy, fig_size=(20, 10), SAVE_PLOTS=True, fil_name=fil_name)
        
        # print(f'# iterations of policy improvement: {len(delta_history)}')
        # print(f'delta = {delta_history}')

        if save_plots is True:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
            ax.plot(np.arange(len(delta_history)) + 1, delta_history, marker='o', markersize=4,
                    alpha=0.7, color='#2ca02c', label=r'$\gamma= $' + f'{self.gamma}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Delta')
            ax.legend()
            plt.tight_layout()
            fil_name = path.join(save_dir, f"value_changes.png")
            plt.savefig(fil_name)
            plt.close()
            