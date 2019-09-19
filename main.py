import gym_super_mario_bros
import tensorflow as tf
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from modules.dqn import CDQN
from modules.preprocess import preprocess_image

def main():
    """
    Main entry point function for program.
    """

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    action_size = len(SIMPLE_MOVEMENT)
    cdqn = CDQN(action_size, memory_size=1000, image_shape=(45, 64, 1))

    batch_size = 32
    games = 1000
    skip = 5

    for game in range(games):

        print("Game: {}".format(game))
        done = True
        for step in range(5000):

            # Skip frames for speed
            if step % skip == 0:
                continue

            # Preprocess first image
            if done:
                state = env.reset()
                state = preprocess_image(state)[..., tf.newaxis]

            # Play move
            action = cdqn.act(state)
            next_state, reward, done, info = env.step(action)

            # Remember move
            next_state = preprocess_image(next_state)[..., tf.newaxis]
            cdqn.remember(state, action, reward, next_state, done)
            state = next_state

            # Render game
            env.render()

            if done:
                cdqn.update_target_model()

            # Train when there are enough examples in memory
            if len(cdqn.memory) >= batch_size:
                cdqn.experience_replay(batch_size)

        tf.saved_model.save(cdqn.network, "model.sav")

    env.close()

if __name__ == "__main__":
    main()
