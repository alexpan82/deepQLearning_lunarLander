import numpy as np
from dqn import QLearningAgent
from sys import argv

if __name__ == '__main__':
    script, command = argv

    if command == "train":

        agent = QLearningAgent(alpha=0.00005, gamma=0.99,
                               epsilon_init=1, epsilon_fin=0.1, epsilon_frame=1000,
                               replay_size=200000, replay_start=50, n_episodes=1000,
                               lag=4, nn_update_freq=1000)
        episode_list, q_state_list, reward_list, frame_counter = agent.solve()

        # Saving rewards and q-values of each episode
        parameters = [frame_counter, agent.epsilon_frame, agent.replay_size, agent.alpha,
                      agent.nn_update_freq, agent.batch_size]
        arr = np.array([episode_list, q_state_list, reward_list, parameters], dtype=object)
        np.save('metadata.npy', arr)

    elif command == "play":
        agent = QLearningAgent(model_path='./q_model.h5')
        agent.play()
