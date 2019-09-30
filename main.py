import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym_super_mario_bros
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

from modules.dqn import CDQN
from modules.preprocess import preprocess_image

def main():
    """
    Main entry point function for program.
    """

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, RIGHT_ONLY)

    action_size = len(RIGHT_ONLY)
    cdqn = CDQN(action_size, memory_size=10000, image_shape=(45, 64, 1))

    batch_size = 1024
    games = 10000
    skip = 100
    beaten = False

    for game in range(games):

        print("Game: {}".format(game + 1), end=" ")
        done = True
        total_reward = 0
        for step in range(8000):

            # Preprocess first image
            if done:
                state = env.reset()
                state = preprocess_image(state)[..., tf.newaxis]

            # Play move
            action = cdqn.act(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Remember move
            next_state = preprocess_image(next_state)[..., tf.newaxis]
            cdqn.remember(state, action, total_reward, next_state, done)
            state = next_state

            # Render game
            env.render()

            if done:
                break

            # Train when there are enough examples in memory
            #if len(cdqn.memory) >= batch_size and step % skip == 0:
        print("Reward: {}".format(total_reward))

        for e in range(5):
            print('Epoch {}'.format(e + 1))
            cdqn.experience_replay(batch_size)

        if game % 10 == 0:
            cdqn.update_target_model()

        print("Reward: {}".format(total_reward))
        tf.saved_model.save(cdqn.network, "model.sav")

    env.close()

if __name__ == "__main__":
    main()
