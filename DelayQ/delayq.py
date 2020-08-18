import random
import numpy as np
from utils.stats import gather_stats

import matplotlib.pyplot as plt
import pyformulas as pf


class DELAYQ(object):
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
        self.m = 30.
        self.e = 1.0

        # Create Table
        shap = [self.groups for i in range(self.state_dim)]
        shap.append(self.action_dim)
        shap = tuple(shap)
        self.Q = np.zeros(shape=shap)
        self.U = np.zeros(shape=shap)
        self.l = np.zeros(shape=shap)
        self.b = np.zeros(shape=shap)
        self.learn = np.full(shap, True)

        # Live Plot Update
        if args.plot:
            self.fig = plt.figure()
            canvas = np.zeros((480, 640))
            self.screen = pf.screen(canvas, "Agent")

    def continous_2_descrete(self, obj, obs):
        """
            Convert Observation to Discrete
        :param obs: env state
        :return: transformed obs
        """
        for i in range(len(obs)):
            obj = obj[int(obs[i] * 100) % self.groups]

        return obj

    def adjust(self, obj, obs, value):
        temp = []
        for i in range(len(obs)):
            v = int(obs[i] * 100) % self.groups
            temp.append(v)
        obs = tuple(temp)

        obj[obs] = value

    def adjustL(self, obs):
        temp = []
        for i in range(len(obs)):
            v = int(obs[i] * 100) % self.groups
            temp.append(v)
        obs = tuple(temp)

        self.l[obs] += 1

    def adjustU(self, obs, r, next_obs):
        temp = []
        for i in range(len(obs)):
            v = int(obs[i] * 100) % self.groups
            temp.append(v)
        obs = tuple(temp)

        self.U[obs] += r + self.gamma * np.argmax(self.continous_2_descrete(self.Q, next_obs))

    def adjustQ(self, obs):
        temp = []
        for i in range(len(obs)):
            v = int(obs[i] * 100) % self.groups
            temp.append(v)
        obs = tuple(temp)

        self.Q[obs] = self.continous_2_descrete(self.U, obs) / self.m + self.e

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        return np.argmax(self.continous_2_descrete(self.Q, s))

    def train(self, env, args):
        results = []
        rewards, rew = [], [-200 for i in range(100)]

        for epoch in range(args.nb_episodes + 1):
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()

            t, t_star = 0, 0
            while not done and time < self.max_steps:
                # Render and Live Plot
                if args.render and epoch % 100 == 0:
                    env.render()
                    if args.plot and time == 0:
                        plt.title("QL - Running Reward")
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

                state_action = np.append(old_state, a)
                if self.continous_2_descrete(self.b, state_action) <= t_star:
                    self.adjust(self.learn, state_action, True)

                if self.continous_2_descrete(self.learn, state_action):
                    if self.continous_2_descrete(self.l, state_action) == 0:
                        self.adjust(self.b, state_action, t)
                    self.adjustL(state_action)
                    self.adjustU(state_action, reward, new_state)

                    if self.continous_2_descrete(self.l, state_action) >= self.m:
                        if self.continous_2_descrete(self.Q, state_action) \
                                - self.continous_2_descrete(self.U, state_action) / self.m >= 2 * self.e:
                            self.adjustQ(state_action)
                            t_star = t
                        elif self.continous_2_descrete(self.b, state_action) > t_star:
                            self.adjust(self.learn, state_action, False)
                        self.adjust(self.U, state_action, 0)
                        self.adjust(self.l, state_action, 0)
                t += 1

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
        path = "./results/delayq"
        self.fig.savefig(path)

    def save_tab(self):
        np.save('./DelayQ/models/q_tab', self.Q)