import numpy as np
import matplotlib.pyplot as plt
from os import path
from copy import deepcopy

class PolicyIteration:
    def __init__(self, problem, gamma, 
                init_policy=None, init_value=None):
        self.problem = problem
        transition_model = problem.transition_model
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.reward_function = np.nan_to_num(problem.reward_function)

        self.transition_model = transition_model
        self.gamma = gamma

        if init_value is None:
            self.values = np.zeros(self.num_states)
        else:
            self.values = init_value
        if init_policy is None:
            self.policy = np.random.randint(0, self.num_actions, self.num_states)
        else:
            self.policy = init_policy
        
        self.policy_history = []
        

    def one_policy_evaluation(self):
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            a = self.policy[s]
            p = self.transition_model[s, a]
            self.values[s] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def run_policy_evaluation(self, tol=1e-3):
        epoch = 0
        delta = self.one_policy_evaluation()
        delta_history = [delta]
        while epoch < 500:
            delta = self.one_policy_evaluation()
            delta_history.append(delta)
            if delta < tol:
                break
        return len(delta_history)

    def run_policy_improvement(self):
        update_policy_count = 0
        for s in range(self.num_states):
            temp = self.policy[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = np.sum(p * self.values)
            self.policy[s] = np.argmax(v_list)
            if temp != self.policy[s]:
                update_policy_count += 1
        return update_policy_count

    def train(self, tol=1e-3, save_plots=True, 
                plot_save_freq=20, 
                save_dir=''):
        epoch = 0
        eval_count_history = []
        policy_change_history = []

        if save_plots:
            self.policy_history = [deepcopy(self.policy)]
            
        while epoch < 500:
            epoch += 1
            eval_count = self.run_policy_evaluation(tol)
            policy_change = self.run_policy_improvement()
            if epoch % plot_save_freq == 0:
                self.policy_history.append(deepcopy(self.policy))
                if save_plots:
                    fil_name = path.join(save_dir, f"policy_epoch{str(epoch).zfill(3)}.png")
                    self.problem.plot_policy(self.policy, fig_size=(20, 10),
                        SAVE_PLOTS=True, fil_name=fil_name)
                
            
            eval_count_history.append(eval_count)
            policy_change_history.append(policy_change)
            if policy_change == 0:
                break
        
        if not epoch % plot_save_freq == 0:
            self.policy_history.append(self.policy)
        if save_plots:
            fil_name = path.join(save_dir, f"policy_epoch_final.png")
            self.problem.plot_policy(self.policy, fig_size=(20, 10), SAVE_PLOTS=True, fil_name=fil_name)
        
        # print(f'# epoch: {len(policy_change_history)}')
        # print(f'eval count = {eval_count_history}')
        # print(f'policy change = {policy_change_history}')

        if save_plots:
            fig, axes = plt.subplots(2, 1, figsize=(3.5, 4), sharex='all', dpi=200)
            axes[0].plot(np.arange(len(eval_count_history)), eval_count_history, marker='o', markersize=4, alpha=0.7,
                         color='#2ca02c', label='# sweep in \npolicy evaluation\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[0].legend()

            axes[1].plot(np.arange(len(policy_change_history)), policy_change_history, marker='o',
                         markersize=4, alpha=0.7, color='#d62728',
                         label='# policy updates in \npolicy improvement\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            plt.tight_layout()
            fil_name = path.join(save_dir, f"sweep_updates.png")
            plt.savefig(fil_name)
            plt.close()
