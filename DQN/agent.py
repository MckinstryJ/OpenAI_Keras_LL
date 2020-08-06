import sys
import numpy as np
import keras.backend as K
from keras import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, Lambda
from keras.activations import relu, linear
from collections import deque
import random


class Agent:
    """ Agent Class (Network) for DQN
    """

    def __init__(self, state_dim, action_dim, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = .99  # Discount Factor
        self.batch_size = 64  # Training Batch Size
        self.lr = lr # Learning Rate
        self.epsilon = 1.0  # Exploration Rate
        self.epsilon_min = .01  # Min to Exploration
        self.epsilon_decay = .005  # Exploration Decay

        # Initialize Deep Q-Network
        self.model = self.network()
        self.model.compile(Adam(lr), 'mse')
        self.memory = deque(maxlen=100000)  # Memory limit of stored states

    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def network(self):
        """ Build Deep Q-Network
        """
        inp = Input((self.state_dim))

        x = Flatten()(inp)
        x = Dense(150, activation='relu')(x)
        x = Dense(120, activation='relu')(x)
        x = Dense(self.action_dim, activation='linear')(x)

        return Model(inp, x)

    def predict(self, inp):
        """ Q-Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def reshape(self, x):
        if len(x.shape) < 4 and len(self.state_dim) > 2: return np.expand_dims(x, axis=0)
        elif len(x.shape) < 3: return np.expand_dims(x, axis=0)
        else: return x

    def save(self, path):
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
