from random import random, randrange, choice
import numpy as np
from utils.stats import gather_stats

import matplotlib.pyplot as plt
import pyformulas as pf


class DYNAQ(object):
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
        self.Q = np.zeros(shape=shap)

        # Memory for Halu
        self.M = []
        # Default or Plus
        self.plus = args.plus
        self.k = 1.0

        # Live Plot Update
        if args.plot:
            self.fig = plt.figure()
            canvas = np.zeros((480, 640))
            self.screen = pf.screen(canvas, "Agent")

    def continous_2_descrete(self, obs):
        """
            Convert Observation to Discrete
        :param obs: env state
        :return: transformed obs
        """

        location = self.Q
        for i in range(len(obs)):
            value = int(obs[i] * 100) % self.groups
            location = location[value]

        return location

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.continous_2_descrete(s))

    def update(self, action, obs, next_obs, reward, last=None, halu=False):
        q_obs = self.continous_2_descrete(obs)
        q_next_obs = self.continous_2_descrete(next_obs)

        q_obs[action] = (1 - self.lr) * q_obs[action] + self.lr * (reward + self.gamma * max(q_next_obs))

        if not halu:
            self.M.append([action, obs, next_obs, reward, 0, True])
        elif halu and self.plus:
            bonus_reward = reward + self.k * np.sqrt(last)
            q_obs[action] = (1 - self.lr) * q_obs[action] + self.lr * (
                        bonus_reward + self.gamma * max(q_next_obs))

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
                        if args.plus:
                            plt.title("DynaQ Plus - Running Reward")
                        else: plt.title("DynaQ - Running Reward")
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

                self.update(a, old_state, new_state, reward)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                old_state = new_state
                cumul_reward += r
                time += 1

            # +1 to last used metric
            for mem in self.M:
                mem[-2] += 1

            # Approximating Optimal Policy via Random Samples
            for n in range(args.hallucinations):
                rand_update = choice(self.M)
                self.update(*rand_update)

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
        if self.plus:
            path = "./results/dynaqp"
        else:
            path = "./results/dynaq"
        self.fig.savefig(path)

    def save_tab(self):
        if self.plus:
            np.save('./DynaQ/models/qplus_tab', self.Q)
        else:
            np.save('./DynaQ/models/q_tab', self.Q)