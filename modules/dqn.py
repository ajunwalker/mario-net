import numpy as np

from base import BaseNetwork


class DQN(BaseNetwork):

    def __init__(self, action_size: int, memory_size: int, lr: float):
        """
        Initializer for Deep Q Network.

        Args:
            action_size: Number of actions the agent can choose from.
            memory_size: Number of episodes for the network to replay from.
            lr: Learning rate of the network.
        """
        self.action_size = action_size
        self.memory = deque(max_len=memory_size)
        self.lr = lr
        self.network = build_model()
        self.target_network = build_model()


    def build_model(self) -> Sequential:
        """
        Constructs and returns a Convolutional Deep-Q-Network
        """

        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=8, strides=(4,4),
                         padding="valid", activation="relu",
                         input_shape = self._image_shape))

        model.add(Conv2D(filters=64, kernel_size=4, strides=(2,2),
                         padding="valid", activation="relu",
                         input_shape=self._image_shape))

        model.add(Conv2D(filters=64, kernel_size=3, strides=(1,1),
                         padding="valid", activation="relu",
                         input_shape=self._image_shape))

        model.add(Flatten())
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(self.action_size))
        model.compile(loss=Huber(), optimizer=self.optimizer,
                      metrics=["accuracy"])
        return model


    def act(self, state: nparray) -> int:
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
            return self.enviroment.action_space.sample()

        # Exploit
        q_values = self.network.predict(state[..., tf.newaxis])
        return np.argmax(q_values[0])


    def remember(self, state: np.array, action: int, reward: float,
                  next_state: np.array, done: boolean) -> None:
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


    def experience_replay(self) -> None:
        """
        Randomly samples a subset of the memory and trains the network with
        the sampled dataset.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.network.predict(state[..., tf.newaxis])
            if done:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state[..., tf.newaxis])
                target[0][action] = reward + self.gamma * np.max(t)
            self.network.fit(state, target, epochs=1, verbose=0)


    def update_target_model(self):
        """
        Copies the weights of the target network with the current agents
        network weights.
        """
        self.target_network.set_weights(self.network.get_weights())
