# Applied Reinforcement Learning in Keras (if needed)

Modular Implementation of all known Reinforcement Learning algorithms in Keras (if needed):

- [x] Q Learning ([QL](https://github.com/MckinstryJ/OpenAI_RL#q-learning-ql))
- [x] State Action Reward State Action ([SARSA](https://github.com/MckinstryJ/OpenAI_RL#state-action-reward-state-action-sarsa))
- [x] Dynamic Q Learning ([DynaQ](https://github.com/MckinstryJ/OpenAI_RL#dynamic-q-learning-dynaq))
- [x] Dynamic Q Learning Plus ([DynaQ+](https://github.com/MckinstryJ/OpenAI_RL#dynamic-q-learning-dynaq))
- [x] Explicit Explore Exploit ([E3](https://github.com/MckinstryJ/OpenAI_RL#explicit-explore-exploit-e3))
- [x] Delayed Q Learning ([DelayQ](https://github.com/MckinstryJ/OpenAI_RL#delay-q-learning-delayq))
- [x] Double Q Learning ([DoubleQ](https://github.com/MckinstryJ/OpenAI_RL#double-q-learning-double-q))
- [x] Actor Critic ([AC](https://github.com/MckinstryJ/OpenAI_RL#actor-critic))
- [x] Synchronous N-step Advantage Actor Critic ([A2C](https://github.com/MckinstryJ/OpenAI_RL#n-step-advantage-actor-critic-a2c))
- [x] Asynchronous N-step Advantage Actor-Critic ([A3C](https://github.com/MckinstryJ/OpenAI_RL#n-step-asynchronous-advantage-actor-critic-a3c))
- [x] Deep Deterministic Policy Gradient with Parameter Noise ([DDPG](https://github.com/MckinstryJ/OpenAI_RL#deep-deterministic-policy-gradient-ddpg))
- [x] Deep Q Network ([DQN](https://github.com/MckinstryJ/OpenAI_RL#deep-q-network-dqn))
- [x] Double Deep Q-Network ([DDQN](https://github.com/MckinstryJ/OpenAI_RL#double-deep-q-network-ddqn))
- [x] Double Deep Q-Network with Prioritized Experience Replay  ([DDQN + PER](https://github.com/MckinstryJ/OpenAI_RL#double-deep-q-network-ddqn))
- [x] Dueling DDQN ([D3QN](https://github.com/MckinstryJ/OpenAI_RL#dueling-double-deep-q-network-dueling-ddqn))

## Getting Started

This implementation requires keras 2.1.6, as well as OpenAI gym.
``` bash
$ pip install gym keras==2.1.6
```

# Tabular Algorithms
### Q Learning (QL)
The Q Learning algorithm is a off-policy method that stores its state-action-rewards in a matrix. The states (for most) come in as continuous so to convert them to discrete numbers, which is used as the index in its reward matrix, the transformation is: 

```python 
int(value * 100) % self.groups
``` 

Where self.groups is the number of "bins". There is a major downside to this transformation, within a smaller grid, because the starting point could easily be seen as the ending point so the action an agent would take in the begining would be effected by what action is needed to gain the highest reward at the end. Still trying to figure out what would be better for these tabular based algorithms that isn't specific to one environment.

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type QL --env LunarLander-v2 --consecutive_frames 1 --plot --render
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     --consecutive_frames 1      |          -273.138 |

### State Action Reward State Action (SARSA)
The SARSA algorithm is an on-policy method that also stores its state-action-rewards in a matrix. The difference between on-policy vs off-policy is whether or not the agent looks at the current policy to determine its discounted reward or some other method (i.e. greedy) 

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type SARSA --env LunarLander-v2 --consecutive_frames 1 --plot --render
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     --consecutive_frames 1      |          -251.316 |

### Dynamic Q Learning (DynaQ)
The Dynamic Q Learning algorithm is an off-policy tabular method that introduces the idea of hallucinations. Before DynaQ, an agent would have to directly interact with the environment to gain insight on what it needs to do next. With hallucinations, however, the agent can revisit states to reinforce the action that produced the given reward.

#### Dynamic Q Learning (Plus)
The plus modification gives a bonus reward to any state that hasn't been revisited in a long time. The bonus is calculated as:
```python
reward + self.k * np.sqrt(last)
```
Where k is manually defined and last is the number of steps since it was last visited.

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type DynaQ --env LunarLander-v2 --consecutive_frames 1 --plot --render --plus
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     --consecutive_frames 1      |          -250.724 |
| Lunar Lander v2         |     --consecutive_frames 1 --plus      |          -144.569 |

### Explicit Explore Exploit (E3)
The E3 algorithm is an off-policy tabular method where the agent will first perform "balanced-wandering" (each action has been selected the same amount) until all actions, in that state, have been selected N times. Once all actions have been selected N times, the state will move into a new list where the standard Q learning method applies. This gives the appearance of explicit exploration and explicit exploitation.

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type E3 --env LunarLander-v2 --consecutive_frames 1 --plot --render
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     --consecutive_frames 1      |          -185.092 |

### Delay Q Learning (DelayQ)
The Delay Q Learning algorithm (aka PAC-MDP) is an off-policy method that acts greedy unless it considers an update based on whether or not the reinforced reward is:
```python
Q[s,a] - U[s,a] / m >= 2 * e
```
Where U is the table for attempted updates and e is a defined value which is the sample complexity of exploration. This is considered as a PAC-MDP because the sample complexity is of the algorithm is less than some polynomial (specified in its paper). 

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type DelayQ --env LunarLander-v2 --consecutive_frames 1 --plot --render
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     --consecutive_frames 1      |          -127.451 |

### Double Q Learning (DoubleQ)
The Double Q Learning algorithm is an off-policy method that employs two reward tables. The update to one uses a discounted value from the other. More specifically, the update to each is:
```python
Qa[s,a] = Qa[s,a] + alpha * (reward + gamma * Qb[s',a] - Qa[s,a]
```

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type DoubleQ --env LunarLander-v2 --consecutive_frames 1 --plot --render
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     --consecutive_frames 1      |          -189.879 |

# Actor-Critic Algorithms
### Actor Critic (AC)
The Actor-Critic algorithm is a model-free, off-policy method where the critic acts as a value-function approximator, and the actor as a policy-function approximator. When training, the critic predicts the TD-Error and guides the learning of both itself and the actor. 

At the present, the AC model is training via CPU and will need to be adjusted to utilize the ```--gpu``` arg. Due to the use of the CPU, the model is very slow.

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type AC --env LunarLander-v2 --gpu --plot --render
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     ---      |          --- |

### N-step Advantage Actor Critic (A2C)
The A2C algorithm approximates the TD-Error using the Advantage funciton. For more stability, we use a shared computational backbone across both networks, as well as an N-step formulation of the discounted rewards. We also incorporate an entropy regularization term ("soft" learning) to encourage exploration. While A2C is simple and efficient, running it on Atari Games quickly becomes intractable due to long computation time.

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type A2C --env LunarLander-v2 --gpu --plot --render
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     ---      |          -116.524 |

### N-step Asynchronous Advantage Actor Critic (A3C)
In a similar fashion as the A2C algorithm, the implementation of A3C incorporates asynchronous weight updates, allowing for much faster computation. We use multiple agents to perform gradient ascent asynchronously, over multiple threads. We test A3C on the Atari Breakout environment.

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type A3C --env LunarLander-v2 --gpu --plot --render
```

You can use ```--render``` but, at this moment, it will render for every thread. Due to this, rendering the environment will slow down training.

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     ---      |          168.795 |

### Deep Deterministic Policy Gradient (DDPG)
The DDPG algorithm is a model-free, off-policy algorithm for continuous action spaces. Similarly to A2C, it is an actor-critic algorithm in which the actor is trained on a deterministic target policy, and the critic predicts Q-Values. In order to reduce variance and increase stability, we use experience replay and separate target networks. Moreover, as hinted by [OpenAI](https://blog.openai.com/better-exploration-with-parameter-noise/), we encourage exploration through parameter space noise (as opposed to traditional action space noise). 

#### Running
Running this algorithm with the full args is shown below:
```bash
$ python3 main.py --type DDPG --env LunarLander-v2 --gpu --plot --render
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     ---      |          -382.363 |

# Deep Q-Learning Algorithms
### Deep Q Network (DQN)
The DQN algorithm is a Q-learning algorithm, which uses a Deep Neural Network as a Q-value function approximator. We estimate target Q-values by leveraging the Bellman equation, and gather experience through an epsilon-greedy policy. For more stability, we sample past experiences randomly (Experience Replay).

### Double Deep Q-Network (DDQN)
A variant of the DQN algorithm is the Double-DQN (or DDQN). For a more accurate estimation of our Q-values, we use a second network to temper the overestimations of the Q-values by the original network. This _target_ network is updated at a slower rate Tau, at every training step.

#### Double Deep Q-Network with Prioritized Experience Replay (DDQN + PER)
We can further improve our DDQN algorithm by adding in Prioritized Experience Replay (PER), which aims at performing importance sampling on the gathered experience. The experience is ranked by its TD-Error, and stored in a SumTree structure, which allows efficient retrieval of the _(s, a, r, s')_ transitions with the highest error.

#### Dueling Double Deep Q-Network (Dueling DDQN)
In the dueling variant of the DQN, we incorporate an intermediate layer in the Q-Network to estimate both the state value and the state-dependent advantage function. After reformulation (see [ref](https://arxiv.org/pdf/1511.06581.pdf)), it turns out we can express the estimated Q-Value as the state value, to which we add the advantage estimate and subtract its mean. This factorization of state-independent and state-dependent values helps disentangling learning across actions and yields better results.

#### Running
Running the algorithm with full args is shown below:
```bash
$ python3 main.py --type DDQN --env CartPole-v1 --batch_size 64
$ python3 main.py --type DDQN --env CartPole-v1 --batch_size 64 --with_PER
$ python3 main.py --type DDQN --env CartPole-v1 --batch_size 64 --dueling
```

#### Results
| Environment &nbsp; &nbsp; &nbsp; &nbsp; | Specific Args | Score |
| :---         |     :---      |          :--- |
| Lunar Lander v2         |     ---      |          245.02 |
| Lunar Lander v2         |     --with_PER      |          99.027 |
| Lunar Lander v2         |     --dueling      |          175.917 |

### Arguments

| Argument &nbsp; &nbsp; &nbsp; &nbsp; | Description | Values |
| :---         |     :---      |          :--- |
| --type         |     Type of RL Algorithm to run      |  Choose from {A2C, A3C, DDQN, DDPG} |
| --env     | Specify the environment       | BreakoutNoFrameskip-v4 (default)      |
| --nb_episodes   | Number of episodes to run     | 5000 (default)    |
| --batch_size     | Batch Size (DDQN, DDPG)  | 32 (default)      |
| --consecutive_frames     | Number of stacked consecutive frames       | 4 (default)      |
| --is_atari     | Whether the environment is an Atari Game with pixel input   | -     |
| --with_PER     | Whether to use Prioritized Experience Replay (with DDQN)      | -      |
| --dueling     | Whether to use Dueling Networks (with DDQN)      | -      |
| --n_threads     | Number of threads (A3C)       | 16 (default)      |
| --gather_stats     | Whether to compute stats of scores averaged over 10 games (slow, see below)       | -      |
| --render     | Whether to render the environment as it is training       | -      |
| --gpu     | GPU index (0)       | -      |
| --plus     | Adds Priority for DynaQ       | -      |
| --clipped     | Convert Reward to -100 or 100       | -      |
| --hallucinations     | Number of inexpensive revisits       | 1000 (default)      |
| --plot     | Live Plot visualization       | -      |

# Visualization & Monitoring

### Model Visualization
All models are saved under ```<algorithm_folder>/models/``` when finished training. You can visualize them running in the same environment they were trained in by running the ```load_and_run.py``` script. For DQN models, you should specify the path to the desired model in the ```--model_path``` argument. For actor-critic models, you need to specify both weight files in the ```--actor_path``` and ```--critic_path``` arguments.

### Tensorboard monitoring
Using tensorboard, you can monitor the agent's score as it is training. When training, a log folder with the name matching the chosen environment will be created. For example, to follow the A2C progression on CartPole-v1, simply run:
```bash
$ tensorboard --logdir=A2C/tensorboard_CartPole-v1/
```
### Results plotting
When training with the argument`--gather_stats`, a log file is generated containing scores averaged over 10 games at every episode: `logs.csv`. Using [plotly](https://plot.ly/), you can visualize the average reward per episode.
To do so, you will first need to install plotly and get a [free licence](https://plot.ly/python/getting-started/).
```bash
pip3 install plotly
```
To set up your credentials, run:
```python
import plotly
plotly.tools.set_credentials_file(username='<your_username>', api_key='<your_key>')
```
Finally, to plot the results, run:
```bash
python3 utils/plot_results.py <path_to_your_log_file>
```

#### Live Plot Visual
When training with the argument `--plot`, a live plot visual will appear besides the render screen. To use this feature you must have `--render` as part of your arguments. The plots are similar to what you might find in results folder. The plot is updated at every 1/10 episodes.

# Acknowledgments

- Atari Environment Helper Class template by [@ShanHaoYu](https://github.com/ShanHaoYu/Deep-Q-Network-Breakout/blob/master/environment.py)
- Atari Environment Wrappers by [OpenAI](github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)
- SumTree Helper Class by [@jaara](https://github.com/jaara/AI-blog/blob/master/SumTree.py)
- Germain and Crew for the base code [@Germain](https://github.com/germain-hug/Advanced-Deep-RL-Keras)

# References (Papers)

- [Delayed Q (PAC-MDP) (DelayQ)](http://cseweb.ucsd.edu/~ewiewior/06efficient.pdf)
- [Advantage Actor Critic (A2C)](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
- [Asynchronous Advantage Actor Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](http://proceedings.mlr.press/v32/silver14.pdf)
- [Hindsight Experience Replay (HER)](https://arxiv.org/pdf/1707.01495.pdf)
- [Deep Q-Learning (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Double Q-Learning (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)
- [Prioritized Experience Replay (PER)](https://arxiv.org/pdf/1511.05952.pdf)
- [Dueling Network Architectures (D3QN)](https://arxiv.org/pdf/1511.06581.pdf)
