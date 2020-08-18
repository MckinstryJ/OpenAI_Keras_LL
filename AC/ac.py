from random import random, randrange
import numpy as np
from utils.networks import tfSummary
from utils.stats import gather_stats
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import pyformulas as pf


class ActorCritic(object):
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

        # Create Model
        num_hidden = 128
        inputs = layers.Input(shape=(self.state_dim,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(self.action_dim, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

        # Live Plot Update
        if args.plot:
            self.fig = plt.figure()
            canvas = np.zeros((480, 640))
            self.screen = pf.screen(canvas, "Agent")

    def train(self, env, args, summary_writer):
        results = []
        rewards, rew = [], [-200 for i in range(100)]
        running_reward = [0]

        # Model fields
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        _loss = keras.losses.squared_hinge
        action_probs_history = []
        critic_value_history = []
        eps = np.finfo(np.float32).eps.item()

        # Main Loop
        tqdm_e = tqdm(range(args.nb_episodes + 1), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()

            with tf.GradientTape() as tape:
                while not done and time < self.max_steps:
                    # Render and Live Plot
                    if args.render and e % 100 == 0:
                        env.render()
                        if args.plot and time == 0:
                            plt.title("AC - Running Reward")
                            plt.ylabel("Reward")
                            plt.plot([np.average(rew[i:i + 100]) for i in range(len(rew) - 100)])
                            self.fig.canvas.draw()
                            image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                            self.screen.update(image)

                    old_state = tf.expand_dims(tf.convert_to_tensor(old_state), 0)
                    # Predicting action probabilities and estimated future rewards
                    action_probs, critic_value = self.model(old_state)
                    critic_value_history.append(critic_value[0, 0])

                    # Sample action from action probability dist.
                    action_probs = tf.keras.backend.eval(action_probs)[0]
                    action = np.random.choice(self.action_dim,
                                              p=action_probs)
                    action_probs_history.append(tf.math.log(action_probs[action]))

                    # Apply the sampled action in our environment
                    new_state, r, done, _ = env.step(action)

                    # Clipping Rewards
                    reward = r
                    if args.clip:
                        reward = np.sign(r) * 100

                    rewards.append(reward)

                    old_state = new_state
                    cumul_reward += r
                    time += 1

                # Update running reward to check if solved
                running_reward.append(0.05 * cumul_reward + (1 - 0.05) * running_reward[-1])

                # Calculate expected value from rewards
                returns = []
                discounted_sum = 0
                for r in rewards[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(action_probs_history, critic_value_history, returns)
                actor_loss = []
                critic_loss = []
                for log_prob, value, ret in history:
                    diff = ret - value
                    actor_loss.append(-log_prob * diff)

                    critic_loss.append(
                        _loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backprop
                loss_value = sum(actor_loss) + sum(critic_loss)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Clear the loss and reward history
                action_probs_history.clear()
                critic_value_history.clear()
                rewards.clear()

            # Gather stats every episode for plotting
            if (args.gather_stats):
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

    def save_image(self):
        path = "./results/ac"
        self.fig.savefig(path)

    def save_tab(self):
        np.save('./AC/models/ac', self.model)