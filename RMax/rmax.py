from random import random, randrange
import numpy as np
from utils.stats import gather_stats
from utils.transforms import *

import matplotlib.pyplot as plt
import pyformulas as pf

'''
    RMAX Algorithm
    - TODO: Need to fix the KeyError (placed on hold for now)
'''

class RMax(object):
    def __init__(self, action_space, state_space, args, group=5):
        self.state_dim = state_space
        self.action_dim = action_space
        self.groups = group
        self.nb_epi = args.nb_episodes

        self.gamma = .995
        self.m = 2.0
        self.rmax = 300

        self.lr = 0.01  # Need?
        self.epsilon = 1.0
        self.epsilon_decay = 1 - (1 / args.nb_episodes)
        self.epsilon_min = .01

        self.max_steps = args.max_steps

        # Create Tables
        shap = [self.groups for i in range(self.state_dim)]
        shap.append(self.action_dim)
        shap = tuple(shap)

        self.T = Transforms(self.nb_epi, self.action_dim)

        self.Q = {}
        self.rsa = {}
        self.n = {}
        self.trans = {}

        # Live Plot Update
        if args.plot:
            self.fig = plt.figure()
            canvas = np.zeros((480, 640))
            self.screen = pf.screen(canvas, "Agent")

    def policy_action(self, s):
        """ Apply an epsilon-greedy policy to pick next action
        """
        obs = self.T.continuous_2_dict(self.Q, s)
        return np.argmax(self.Q[obs])

    def update(self, action, obs, next_obs, reward):

        if action is not None and obs is not None:
            q_obs = self.T.continuous_2_dict(self.Q, obs)

            rew_obs = self.T.continuous_2_dict(self.rewards, obs)
            rsa_obs = self.T.continuous_2_dict(self.rsa_counts, obs)
            trans_obs = self.T.continuous_2_dict(self.transitions, obs)
            tsa_obs = self.T.continuous_2_dict(self.tsa_counts, obs)

            self.rewards[rew_obs][action] += reward
            self.rsa_counts[rsa_obs][action] += 1
            self.transitions[trans_obs][action] += 1
            self.tsa_counts[tsa_obs][action] += 1

            if self.rsa_counts[rsa_obs][action] == self.state_action_thres:
                lim = int(np.log(1 / (self.epsilon * (1 - self.gamma))) / (1 - self.gamma))
                for i in range(1, lim):
                    for current_state in self.rewards.keys():
                        for current_action in range(self.action_dim):
                            if self.rsa_counts[current_state][current_action] >= self.state_action_thres:
                                rewards_s_a = self.rewards[current_state]

                                self.Q[current_state][current_action] = np.average(rewards_s_a)
                            else:
                                self.Q[current_state][current_action] = self.rmax

    def train(self, env, args):
        results = []
        rewards, rew = [], [-200 for i in range(100)]

        for epoch in range(args.nb_episodes + 1):
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()

            while not done and time < self.max_steps:
                # Render and Live Plot
                if args.render and epoch % 100 == 0:
                    env.render()
                    if args.plot and time == 0:
                        plt.title("RMax - Running Reward")
                        plt.ylabel("Reward")
                        plt.plot([np.average(rew[i:i + 100]) for i in range(len(rew) - 100)])
                        self.fig.canvas.draw()
                        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                        self.screen.update(image)

                a = self.policy_action(old_state)
                new_state, r, done, _ = env.step(a)

                # R Max - Main Part
                s = self.T.continuous_2_dict(self.n, old_state)
                if self.n[s][a] < self.m:
                    self.T.continuous_2_dict(self.rsa, old_state)
                    ssp = self.T.continuous_2_dict(self.trans, old_state + new_state)

                    self.n[s][a] += 1
                    self.rsa[s][a] += r
                    self.trans[ssp][a] += 1

                    if self.n[s][a] == self.m:
                        lim = int(np.log(1 / (self.epsilon * (1 - self.gamma))) / (1 - self.gamma))
                        for i in range(1, lim):
                            for current_state in self.n.keys():
                                cs = self.T.continuous_2_dict(self.Q, current_state)
                                for current_action in range(self.action_dim):
                                    if self.n[current_state][current_action] >= self.m:
                                        ns = self.T.continuous_2_dict(self.Q, new_state)

                                        self.Q[cs][current_action] += self.rsa[cs][current_action] \
                                            + self.gamma * (np.sum(self.trans[cs + ns][current_action]) / self.n[cs][current_action]) \
                                            * np.max(self.Q[ns])

                # Clipping Rewards
                reward = r
                if args.clip:
                    reward = np.sign(r) * 100

                old_state = new_state
                cumul_reward += r
                time += 1

            # Gather stats every episode for plotting
            rewards.append(cumul_reward)
            rew.append(cumul_reward)
            if args.gather_stats:
                mean, stdev = gather_stats(self, env)
                results.append([epoch, mean, stdev])

            # Display score
            if epoch % 100 == 0:
                s = round(np.average(rew[-100:]), 5)
                print("Epoch - {} ---> Score - {}".format(epoch, s))

        if args.plot:
            self.save_image()

        return results

    def save_image(self):
        path = "./results/q"
        self.fig.savefig(path)

    def save_tab(self):
        np.save('./QL/models/q_tab', self.Q)