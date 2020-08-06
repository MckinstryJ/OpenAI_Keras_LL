from random import random, randrange

from utils.memory_buffer import MemoryBuffer
import numpy as np
from tqdm import tqdm

from DQN.agent import Agent
from utils.networks import tfSummary
from utils.stats import gather_stats

import matplotlib.pyplot as plt
import pyformulas as pf


class DQN:

    def __init__(self, action_dim, state_dim, args):
        self.action_dim = action_dim
        self.state_dim = (args.consecutive_frames,) + state_dim

        self.max_steps = args.max_steps # default 3000
        self.with_per = args.with_per
        self.buffer_size = 100000

        self.gamma = .99  # Discount Factor
        self.batch_size = 64  # Training Batch Size
        self.lr = .0005  # Learning Rate
        self.epsilon = 1.0  # Exploration Rate
        self.epsilon_min = .01  # Min to Exploration
        self.epsilon_decay = .005  # Exploration Decay

        # Create networks
        self.agent = Agent(self.state_dim, action_dim, self.lr)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, args.with_per)

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
            return np.argmax(self.agent.predict(s)[0])

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        s, a, r, d, next_s, idx = self.buffer.sample_batch(batch_size)

        s = np.squeeze(s)  # squeeze combines all to a single dimension
        next_s = np.squeeze(next_s)

        # Target value = reward + discount * (max action -> reward on next states)
        targets = r + self.gamma * (np.amax(self.agent.model.predict_on_batch(next_s), axis=1)) * (1 - d)
        # Target value = predicted reward for current states
        targets_full = self.agent.model.predict_on_batch(s)
        # For every index in batch, adjust rewards of current as a portion of the rewards in the future
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [a]] = targets

        # Retraining Model with states and rewards
        self.agent.fit(s, targets_full)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def train(self, env, args, summary_writer):
        results = []
        rewards, rew = [], [-200 for i in range(100)]
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()

            while not done and time < self.max_steps:
                # Render and Live Plot
                if args.render and e % 100 == 0:
                    env.render()
                    if args.plot and time == 0:
                        plt.title("DQN - Running Reward")
                        plt.ylabel("Reward")
                        plt.plot([np.average(rew[i:i + 100]) for i in range(len(rew) - 100)])
                        self.fig.canvas.draw()
                        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                        self.screen.update(image)

                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)

                new_state, r, done, _ = env.step(a)
                cumul_reward += r

                # Storing the set for retraining purposes
                self.memorize(old_state, a, r, done, new_state)
                old_state = new_state

                time += 1
                # Train DQN
                if self.buffer.size() > args.batch_size:
                    self.train_agent(args.batch_size)

            # Gather stats every episode for plotting
            rewards.append(cumul_reward)
            rew.append(cumul_reward)
            if args.gather_stats:
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            s = round(np.average(rew[-100:]), 5)
            tqdm_e.set_description("Score: " + str(s))
            tqdm_e.refresh()

        self.save_image()

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        if self.with_per:
            q_val = self.agent.predict(state)
            next_best_action = np.argmax(self.agent.predict(new_state))
            new_val = reward + self.gamma * q_val[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.agent.save(path)

    def save_image(self):
        path = "./results/dqn"
        if self.with_per:
            path += "_with_PER"

        self.fig.savefig(path)
