from collections import deque
from random import randint, sample

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

class CDQN:

    def __init__(self, action_size: int, memory_size: int, image_shape: tuple):
        """
        Initializer for Convolutional Deep-Q-Network.

        Args:
            action_size: Number of actions the agent can choose from.
            memory_size: Number of episodes for the network to replay from.
            lr: Learning rate of the network.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.min_epsilon = 0.0001
        self.epsilon_decay = 0.0001
        self.gamma = 1
        self.image_shape = image_shape
        self.network = self.build_model()
        self.target_network = self.build_model()


    def build_model(self) -> Sequential:
        """
        Constructs and returns a Convolutional Deep-Q-Network
        """
        model = Sequential()

        model.add(Conv2D(filters=8, kernel_size=4, strides=(2,2),
                         padding="valid", activation="relu",
                         input_shape = self.image_shape))

        #model.add(Conv2D(filters=64, kernel_size=4, strides=(2,2),
        #                 padding="valid", activation="relu"))

        #model.add(Conv2D(filters=64, kernel_size=3, strides=(1,1),
        #                 padding="valid", activation="relu"))

        model.add(Flatten())
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(self.action_size))
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=0.01),
                      metrics=["accuracy"])
        return model


    def act(self, state: np.array) -> int:
        """
        Returns an action based on the following rule:
        1. If randomly generated number if below the exploration
           rate (epsilon), select a random action (explore).
        2. If randomly generated number is above the eploration rate,
           predict next action using network (exploit).

        Args:
            state: Array describing the state the agent is in.
        """
        # Explore
        if np.random.rand() <= self.epsilon:

            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.epsilon_decay

            return randint(0, self.action_size - 1)

        # Exploit
        state = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
        q_values = self.network.predict(state)
        return np.argmax(q_values[0])


    def remember(self, state: np.array, action: int, reward: float,
                  next_state: np.array, done: bool) -> None:
        """
        Saves the current state of the game for memory replay.

        Args:
            state: Previous state the agent was at.
            action: Action number it took at previous state.
            reward: Reward obtained from taking action.
            next_state: The state it ended up in after taking specified action.
            done: Whether the environment has terminated or not.
        """
        self.memory.append((state, action, reward, next_state, done))


    def experience_replay(self, batch_size) -> None:
        """
        Randomly samples a subset of the memory and trains the network with
        the sampled dataset.

        Args:
            batch_size: Size of subsample.
        """
        minibatch = sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
            target = self.network.predict(state)
            if done:
                target[0][action] = reward
            else:
                next_state = np.expand_dims(np.asarray(next_state).astype(np.float64), axis=0)
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.max(t)
            self.network.fit(state, target, epochs=1, verbose=0)


    def update_target_model(self):
        """
        Copies the weights of the target network with the current agents
        network weights.
        """
        self.target_network.set_weights(self.network.get_weights())
