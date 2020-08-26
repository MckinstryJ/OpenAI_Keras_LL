from random import random, randrange
import numpy as np
from utils.stats import gather_stats
from utils.transforms import *

import matplotlib.pyplot as plt
import pyformulas as pf


class DOUBLEQ(object):
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

        # Create Table
        shap = [self.groups for i in range(self.state_dim)]
        shap.append(self.action_dim)
        shap = tuple(shap)
        self.Qa = {}
        self.Qb = {}

        # Live Plot Update
        if args.plot:
            self.fig = plt.figure()
            canvas = np.zeros((480, 640))
            self.screen = pf.screen(canvas, "Agent")

    def policy_action(self, s):
        """ Apply an epsilon-greedy policy to pick next action
        """
        obs_a = continuous_2_dict(self.Qa, s, self.nb_epi, self.action_dim)
        obs_b = continuous_2_dict(self.Qb, s, self.nb_epi, self.action_dim)
        if max(self.Qa[obs_a]) < max(self.Qb[obs_b]):
            return np.argmax(self.Qb[obs_b])
        return np.argmax(self.Qa[obs_a])

    def update(self, Q, action, obs, next_obs, reward):
        if Q == 0:
            a_star_obs = continuous_2_dict(self.Qa, next_obs, self.nb_epi, self.action_dim)
            a_star = np.argmax(self.Qa[a_star_obs])

            obs_a = continuous_2_dict(self.Qa, obs, self.nb_epi, self.action_dim)
            obs_b = continuous_2_dict(self.Qb, next_obs, self.nb_epi, self.action_dim)
            self.Qa[obs_a][action] = self.Qa[obs_a][action] + self.lr \
                                   * (reward + self.gamma * self.Qb[obs_b][a_star] - self.Qa[obs_a][action])
        else:
            b_star_obs = continuous_2_dict(self.Qb, next_obs, self.nb_epi, self.action_dim)
            b_star = np.argmax(self.Qb[b_star_obs])

            obs_b = continuous_2_dict(self.Qb, obs, self.nb_epi, self.action_dim)
            obs_a = continuous_2_dict(self.Qa, next_obs, self.nb_epi, self.action_dim)

            self.Qb[obs_b][action] = self.Qb[obs_b][action] + self.lr \
                                     * (reward + self.gamma * self.Qa[obs_a][b_star] - self.Qb[obs_b][action])

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
                        plt.title("Double Q - Running Reward")
                        plt.ylabel("Reward")
                        plt.plot([np.average(rew[i:i + 100]) for i in range(len(rew) - 100)])
                        self.fig.canvas.draw()
                        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                        self.screen.update(image)

                a = self.policy_action(old_state)
                new_state, r, done, _ = env.step(a)

                # Clipping Rewards
                reward = r
                if args.clip:
                    reward = np.sign(r) * 100

                self.update(np.random.randint(0, 1), a, old_state, new_state, reward)

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
        path = "./results/dq"
        self.fig.savefig(path)

    def save_tab(self):
        np.save('./DoubleQ/models/qa_tab', self.Qa)
        np.save('./DoubleQ/models/qb_tab', self.Qb)