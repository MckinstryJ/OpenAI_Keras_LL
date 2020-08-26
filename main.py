""" RL Algorithms for OpenAI Gym environments
"""
import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd

from A2C.a2c import A2C
from A3C.a3c import A3C
from DDQN.ddqn import DDQN
from DDPG.ddpg import DDPG

# John's Additions
from DQN.dqn import DQN
from QL.q import QL
from SARSA.sarsa import SARSA
from DynaQ.dynaq import DYNAQ
from DoubleQ.dq import DOUBLEQ
from DelayQ.delayq import DELAYQ
from AC.ac import ActorCritic
from E3.e3 import E3
from Random.rand import Random

import keras.backend.tensorflow_backend as K

from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import get_session

gym.logger.set_level(40)


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DDQN',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    parser.add_argument('--plus', dest='plus', action='store_true', help="Use Dyna Q+ feature (DynaQ)")
    parser.add_argument('--clipped', dest='clip', action='store_true', help="Clip the reward generated to -1 or 1")
    #
    parser.add_argument('--nb_episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    parser.add_argument('--max_steps', type=int, default=3000, help="Max Steps agent takes in environment.")
    parser.add_argument('--hallucinations', type=int, default=1000, help="Number of state/reward revisits.")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true', help="Compute Average reward per episode (slower)")
    parser.add_argument('--plot', dest='plot', action='store_true', help="Update Live Plot at same rate of 'render'")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='LunarLander-v2',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='To Use GPU.')
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        K.set_session(get_session())
        K.tf.local_variables_initializer()
        summary_writer = K.tf.summary.FileWriter(args.type + "/tensorboard_" + args.env)

    # Environment Initialization
    if args.is_atari:
        # Atari Environment Wrapper
        env = AtariEnvironment(args)
        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
    elif args.type== "DDPG":
        # Continuous Environments Wrapper
        env = Environment(gym.make(args.env), args.consecutive_frames)
        env.reset()
        state_dim = env.get_state_size()
        action_space = gym.make(args.env).action_space
        # TODO: adjust for continuous environments
        action_dim = action_space.n
        act_range = action_space.n
    elif args.type in ["QL", "SARSA", "DynaQ", "DoubleQ",
                       "DelayQ", "AC", "E3", "Random"]:
        env = gym.make(args.env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    else:
        # Standard Environments
        env = Environment(gym.make(args.env), args.consecutive_frames)
        env.reset()
        state_dim = env.get_state_size()
        action_dim = gym.make(args.env).action_space.n

    # Pick algorithm to train
    if args.type == "QL":
        algo = QL(action_dim, state_dim, args)
    elif args.type == "SARSA":
        algo = SARSA(action_dim, state_dim, args)
    elif args.type == "DynaQ":
        algo = DYNAQ(action_dim, state_dim, args)
    elif args.type == "DoubleQ":
        algo = DOUBLEQ(action_dim, state_dim, args)
    elif args.type == "DelayQ":
        algo = DELAYQ(action_dim, state_dim, args)
    elif args.type == "AC":
        algo = ActorCritic(action_dim, state_dim, args)
    elif args.type == "E3":
        algo = E3(action_dim, state_dim, args)
    elif args.type == "Random":
        algo = Random(action_dim, state_dim, args)
    elif args.type == "DQN":
        algo = DQN(action_dim, state_dim, args)
    elif args.type == "DDQN":
        algo = DDQN(action_dim, state_dim, args)
    elif args.type == "A2C":
        algo = A2C(action_dim, state_dim, args.consecutive_frames)
    elif args.type == "A3C":
        algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari)
    elif args.type == "DDPG":
        algo = DDPG(action_dim, state_dim, act_range, args.consecutive_frames)

    # Train
    if args.type in ["QL", "SARSA", "DynaQ", "DoubleQ",
                     "DelayQ", "E3", "Random"]:
        stats = algo.train(env, args)
    else:
        stats = algo.train(env, args, summary_writer)

    # Export results to CSV
    if args.gather_stats:
        df = pd.DataFrame(np.array(stats))
        df.plot()
        df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%.5f')

    # Save weights and close environments
    exp_dir = '{}/models/'.format(args.type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}{}_ENV_{}_NB_EP_{}_BS_{}'.format(exp_dir,
        args.type,
        args.env,
        args.nb_episodes,
        args.batch_size)

    if args.type in ["QL", "SARSA", "DynaQ", "DoubleQ",
                     "DelayQ", "E3", "Random"]:
        algo.save_tab()
    else:
        algo.save_weights(export_path)
    env.env.close()


if __name__ == "__main__":
    main()
