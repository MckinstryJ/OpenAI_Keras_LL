from random import random, randrange
import numpy as np
from utils.stats import gather_stats

import matplotlib.pyplot as plt
import pyformulas as pf


class DOUBLEQ(object):
    def __init__(self, action_space, state_space, args, group=5):
        self.state_dim = state_space
        self.action_dim = action_space
        self.groups = group

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
        self.Qa = np.zeros(shape=shap)
        self.Qb = np.zeros(shape=shap)

        # Live Plot Update
        if args.plot:
            self.fig = plt.figure()
            canvas = np.zeros((480, 640))
            self.screen = pf.screen(canvas, "Agent")

    def continous_2_descrete(self, Q, obs):
        """
            Convert Observation to Discrete
        :param obs: env state
        :return: transformed obs
        """

        location = Q
        for i in range(len(obs)):
            value = int(obs[i] * 100) % self.groups
            location = location[value]

        return location

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        actions_a = self.continous_2_descrete(self.Qa, s)
        actions_b = self.continous_2_descrete(self.Qb, s)
        if max(actions_a) < max(actions_b):
            return np.argmax(actions_b)
        return np.argmax(actions_a)

    def update(self, Q, action, obs, next_obs, reward):
        if Q == 0:
            a_star = np.argmax(self.continous_2_descrete(self.Qa, next_obs))
            Qa = self.continous_2_descrete(self.Qa, obs)
            Qb = self.continous_2_descrete(self.Qb, next_obs)
            Qa[action] = Qa[action] + self.lr * (reward + self.gamma * Qb[a_star] - Qa[action])
        else:
            b_star = np.argmax(self.continous_2_descrete(self.Qb, next_obs))
            Qb = self.continous_2_descrete(self.Qa, obs)
            Qa = self.continous_2_descrete(self.Qb, next_obs)
            Qb[action] = Qb[action] + self.lr * (reward + self.gamma * Qa[b_star] - Qb[action])

    def train(self, env, args):
        results = []
        rewards, rew = [], [-200 for i in range(100)]

        for epoch in range(args.nb_episodes + 1 + 4000):
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