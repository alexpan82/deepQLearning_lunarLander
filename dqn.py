# # Learning to Play Lunar Lander Using Deep Q-Learning
# * Adapated from Mnih, Kavukcuoglu, Silver et al
# * RL algorithms are written from scratch!
# * Neural nets manipulated using TensorFlow

import gym
import numpy as np
from neuralNet import deep_q_net
import matplotlib.pyplot as plt


class QLearningAgent(object):
    def __init__(self, alpha, gamma, n_episodes,
                 epsilon_init=1, epsilon_fin=0.1, epsilon_frame=1000,
                 replay_size=200000, replay_start=50, lag=4, model_path='',
                 nn_update_freq=20):

        # Initialize parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.epsilon_init = float(epsilon_init)
        self.epsilon = float(epsilon_init) # initial epsilon-greedy value
        self.epsilon_fin = float(epsilon_fin)   # final epsilon-greedy value
        self.epsilon_frame = float(epsilon_frame)   # Frame at which final epsilon-greedy value is maintained
        self.step_size = (self.epsilon_init - self.epsilon_fin) / self.epsilon_frame
        self.n_episodes = n_episodes    # number of episodes to train
        self.replay_size = replay_size  # number of state transitions to store in memory
        self.replay_start = replay_start    # number of elements in replay before training
        self.lag = lag  # number of frames to lag by (repeat same action for 4 frames) (also dont train for 4 frames)
        self.n_actions = 4  # Number of actions agent can take
        self.shape = 8  # Length on the state vector
        self.batch_size = 32    # Size of state-transition minibatches to train NN with
        self.nn_update_freq = int(nn_update_freq)   # number of steps before copying Q parameters to target
        self.render_freq = 100    # Render the game every n episodes
        self.model_path = model_path
        # Initialize environment with time limit
        self.env = gym.make("LunarLander-v2")
        # self.env = gym.make("CartPole-v0")
        
        # Seed
        np.random.seed(123)
        self.env.seed(123)

        self.test_state = np.load('./test_initial_states.npy', allow_pickle=True)
        # self.test_state = [0, 0, 0, 0]

    def solve(self):
        """ Initialize """
        # Initialize Q(s,a) and target Q'(s,a) neural nets
        dqn_object = deep_q_net(alpha=self.alpha, n_actions=self.n_actions,
                                gamma=self.gamma, n_samples=self.batch_size,
                                shape=self.shape, model_path=self.model_path)
        # Initialize replay buffer (max size will be limited to self.replay_size)
        replay_buffer = np.empty((0, 5), object)
        frame_counter = 0
        parameter_updates = 1
        episode_list = [0]
        reward_list = [0]
        reward_rolling = [np.nan]  # Rolling average of the reward
        episode_rolling = [np.nan]
        q_state_list = [0]  # keep track of q-value for self.test_state
        q_action_list = np.zeros((1, self.n_actions))

        # Plotting Reward
        fig, axs = plt.subplots(2)
        axs[0].set_title("Total Reward")
        axs[1].set_title("Avg Q value for a state")
        li, = axs[0].plot(episode_list, reward_list, label="Reward")
        li1, = axs[0].plot(episode_list, reward_rolling, label="Rolling Average")
        # Plot Average Q values for states along with each the Q value for each action
        li2, = axs[1].plot(episode_list, q_state_list, label="Q(s_i,a)")
        li3, = axs[1].plot(episode_list, q_state_list, label="Q(s_i,0)", linestyle='--')
        li4, = axs[1].plot(episode_list, q_state_list, label="Q(s_i,1)", linestyle='--')
        li5, = axs[1].plot(episode_list, q_state_list, label="Q(s_i,2)", linestyle='--')
        li6, = axs[1].plot(episode_list, q_state_list, label="Q(s_i,3)", linestyle='--')
        axs[0].legend()
        axs[1].legend()
        fig.tight_layout()
        fig.canvas.draw()
        plt.show(block=False)

        """ Implement Q-learning """
        for episode in range(self.n_episodes):
            # Initialize environment
            state = self.env.reset()
            action = 0  # Initial action just for good practice
            total_reward = 0    # Keep track of total score per episode
            while True:
                frame_counter += 1
                # Conditional for lag
                if frame_counter % self.lag == 0:
                    # Choose action using policy derived from Q (epsilon-greedy)
                    # Linearly decay epsilon-greedy policy
                    action = self.e_greedy_action(frame_counter, dqn_object, state)

                    # Take action and observe R, S'
                    state_next, reward, done, info = self.env.step(action)
                    transition_vector = np.array([state, action, reward, state_next, done], dtype=object)

                    # Store transition into replay buffer
                    # Limit size (see method below)
                    replay_buffer, train_bool = self.store_in_buffer(replay_buffer, transition_vector)

                    # Update Q (only after
                    # This method will sample a random batch from the replay_buffer
                    # and will perform a gradient descent step
                    if train_bool:
                        dqn_object.train_q(replay_buffer)

                    # Copy Q network parameters over to the target Q'
                    # every self.nn_update_freq
                    if parameter_updates == self.nn_update_freq:
                        print("Model Copied!")
                        dqn_object.copy_model()
                        parameter_updates = 1
                    else:
                        parameter_updates += 1
                    # Set next state
                    state = state_next

                else:
                    # Do NOT train
                    # action = self.e_greedy_action(frame_counter, dqn_object, state)
                    if frame_counter <= self.epsilon_frame:
                        self.epsilon -= self.step_size
                    else:
                        self.epsilon = self.epsilon_fin
                    state_next, reward, done, info = self.env.step(action)
                    transition_vector = np.array([state, action, reward, state_next, done], dtype=object)
                    replay_buffer, train_bool = self.store_in_buffer(replay_buffer, transition_vector)
                    state = state_next

                total_reward += reward

                if episode % self.render_freq == 0:
                    self.env.render()

                if done:
                    break
            # Output metrics
            avg_per_action, avg_per_state, avg_q_scalar = dqn_object.avg_q_value(self.test_state)

            print("Episode: ", episode, "Reward: ", total_reward, "Frame: ", frame_counter,
                  "Epsilon: ", self.epsilon, "Avg Q-value for test states: ", avg_q_scalar,
                  "Replay buffer size: ", len(replay_buffer))
            self.env.render()
            print("Saving weights")
            dqn_object.save_model()

            # Plot metrics
            episode_list.append(episode+1)
            reward_list.append(total_reward)
            q_state_list.append(avg_q_scalar)
            q_action_list = np.append(q_action_list, np.array([avg_per_action]), axis=0)
            if len(reward_list) >= 50:
                reward_rolling.append(np.mean(reward_list[-50:]))
                episode_rolling.append(episode + 1)
                li1.set_xdata(episode_rolling)
                li1.set_ydata(reward_rolling)
            else:
                reward_rolling.append(np.nan)
                episode_rolling.append(np.nan)

            li.set_xdata(episode_list)
            li.set_ydata(reward_list)

            li2.set_xdata(episode_list) # Sorry!!!!
            li2.set_ydata(q_state_list)
            li3.set_xdata(episode_list)
            li3.set_ydata(q_action_list[:, 0])
            li4.set_xdata(episode_list)
            li4.set_ydata(q_action_list[:, 1])
            li5.set_xdata(episode_list)
            li5.set_ydata(q_action_list[:, 2])
            li6.set_xdata(episode_list)
            li6.set_ydata(q_action_list[:, 3])

            axs[0].relim()
            axs[0].autoscale_view(True, True, True)
            axs[1].relim()
            axs[1].autoscale_view(True, True, True)
            fig.canvas.draw()

            # Quit when rolling average reaches 200
            if reward_rolling[-1] >= 200:
                break

        plt.show()

        return episode_list, q_state_list, reward_list, frame_counter

    def e_greedy_action(self, num_steps, dqn_object, state):
        # Linearly decrease epsilon from epsilon_init to epsilon_fin over the first epsilon_frame frames
        # Then fixed at epsilon_fin thereafter
        # Return the epsilon-greedy action as determined by the target nn
        if num_steps <= self.epsilon_frame:
            self.epsilon -= self.step_size
        else:
            self.epsilon = self.epsilon_fin

        rand = np.random.random_sample()
        if rand < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Choose greediest action
            action = dqn_object.q_predict_action(np.array([state])).numpy()[0]
        return action

    def store_in_buffer(self, replay_buffer, transition_vector):
        # Store transition into replay buffer
        # Make sure not to exceed self.replay_size in length
        # 1st element in replay_buffer is empty, so effectively 1-indexed
        # When self.replay_size is reached, just replace random elements with new transitions
        # Also return a boolean for whether buffer size is at least self.replay_start
        buffer_size = len(replay_buffer)
        replay_buffer = np.append(replay_buffer, [transition_vector], axis=0)
        if buffer_size > self.replay_size:
            replay_buffer = np.delete(replay_buffer, 0, 0)  # delete index 0 along axis 0

        if buffer_size < self.replay_start:
            train_bool = False
        else:
            train_bool = True

        return replay_buffer, train_bool

    def play(self):
        # Plotting Reward
        reward_list = []
        avg_list = []
        episode_list = []
        fig, axs = plt.subplots(1)
        axs.set_title("Total Reward")
        li, = axs.plot(episode_list, reward_list, label="Reward for Episode")
        li1, = axs.plot(episode_list, avg_list, label="Average Reward at Episode")
        leg = axs.legend()
        fig.tight_layout()
        fig.canvas.draw()
        plt.show(block=False)

        """ Initialize """
        # Initialize Q(s,a) and target Q'(s,a) neural nets
        dqn_object = deep_q_net(alpha=self.alpha, n_actions=self.n_actions,
                                gamma=self.gamma, n_samples=self.batch_size,
                                shape=self.shape, model_path=self.model_path)
        # Initialize replay buffer (max size will be limited to self.replay_size)
        frame_counter = 0

        for episode in range(self.n_episodes):
            # Initialize environment
            state = self.env.reset()
            action = 0  # Initial action just for good practice
            total_reward = 0    # Keep track of total score per episode
            while True:
                frame_counter += 1
                # Conditional for lag
                if frame_counter % self.lag == 0:
                    # Choose action using policy derived from Q (greedy)
                    action = dqn_object.q_predict_action(np.array([state])).numpy()[0]

                    # Take action and observe R, S'
                    state_next, reward, done, info = self.env.step(action)

                    # Set next state
                    state = state_next

                else:
                    # Do NOT take greedy action: take last action
                    state_next, reward, done, info = self.env.step(action)
                    state = state_next

                total_reward += reward
                self.env.render()

                if done:
                    print(state)
                    break
            print(total_reward)

            episode_list.append(episode+1)
            reward_list.append(total_reward)
            avg_list.append(np.mean(reward_list))
            li.set_xdata(episode_list)
            li.set_ydata(reward_list)
            li1.set_xdata(episode_list)
            li1.set_ydata(avg_list)
            axs.relim()
            axs.autoscale_view(True, True, True)
            fig.canvas.draw()

        plt.show()
