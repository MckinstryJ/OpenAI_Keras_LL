'''

    Explicit Explore or Exploit (E3)
    via MIT OCW: https://ocw.mit.edu/courses/mechanical-engineering/2-997-decision-making-in-large-scale-systems-spring-2004/lecture-notes/lec_9_v1.pdf

    OpenAI doesn't provide an optimal policy, the agent has to find it.
    So, the algorithm will be slightly modified to fill that gap.
'''
from random import random, randrange
import numpy as np
from utils.stats import gather_stats
from utils.transforms import *

import matplotlib.pyplot as plt
import pyformulas as pf


class E3(object):
    def __init__(self, action_space, state_space, args, group=5):
        self.state_dim = state_space
        self.action_dim = action_space
        self.groups = group
        self.nb_epi = args.nb_episodes

        self.gamma = .995
        self.lr = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 1 - (1 / args.nb_episodes)
        self.epsilon_min = .01
        self.max_steps = args.max_steps
        self.N = []
        self.X = 5

        # Create Table
        shap = [self.groups for i in range(self.state_dim)]
        shap.append(self.action_dim)
        shap = tuple(shap)
        self.Q = {}
        self.B = {}

        # Live Plot Update
        if args.plot:
            self.fig = plt.figure()
            canvas = np.zeros((480, 640))
            self.screen = pf.screen(canvas, "Agent")

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            obs = continuous_2_dict(self.Q, s, self.nb_epi, self.action_dim)
            return np.argmax(self.Q[obs])

    def update(self, action, obs, next_obs, reward):
        obs = continuous_2_dict(self.Q, obs, self.nb_epi, self.action_dim)
        next_obs = continuous_2_dict(self.Q, next_obs, self.nb_epi, self.action_dim)

        self.Q[obs][action] = (1 - self.lr) * self.Q[obs][action] \
                              + self.lr * (reward + self.gamma * max(self.Q[next_obs]))

    def train(self, env, args):
        '''
            1. If state is not in N, perform "balanced wandering" till seen X times
            2. Else If state is in N and reward is greater than THEATA, exploit
            3. Else If state is in N, follow explore policy

        '''

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
                        plt.title("E3 - Running Reward")
                        plt.ylabel("Reward")
                        plt.plot([np.average(rew[i:i + 100]) for i in range(len(rew) - 100)])
                        self.fig.canvas.draw()
                        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                        self.screen.update(image)

                # If state is not in N, "balanced wandering" till all seen X times
                old_s = continuous_2_dict(self.Q, old_state, self.nb_epi, self.action_dim)
                if old_s not in self.N:
                    obs = continuous_2_dict(self.B, old_state, self.nb_epi, self.action_dim)
                    lowest = np.argmin(self.B[obs])
                    self.B[obs][lowest] += 1
                    if self.B[obs][lowest] >= self.X:
                        self.N.append(old_s)
                    a = lowest
                else:
                    a = self.policy_action(old_state)

                new_state, r, done, _ = env.step(a)

                # Clipping Rewards
                reward = r
                if args.clip:
                    reward = np.sign(r) * 100

                self.update(a, old_state, new_state, reward)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

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

        self.save_image()

        return results

    def save_image(self):
        path = "./results/e3"
        self.fig.savefig(path)

    def save_tab(self):
        np.save('./E3/models/q_tab', self.Q)