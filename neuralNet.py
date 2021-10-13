import tensorflow as tf
import numpy as np

class deep_q_net(object):
    def __init__(self, alpha=0.01, n_actions=4, n_samples=32,
                 gamma=0.99, shape=8, model_path=''):
        # Set seed
        tf.random.set_seed(123)
        self.ALPHA = alpha  # learning rate
        self.N_ACTIONS = n_actions  # number of actions agent can take
        self.GAMMA = gamma
        self.N_SAMPLES = n_samples
        self.SHAPE = shape   # Shape of state returned by gym
        self.model_path = model_path

        # If GPU exists, don't use too much memory
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
                
        # Create action-value Q function
        # Just do one hidden layer with 64 nodes to test
        # Can also import h5 file
        if self.model_path == '':
            self.q_model = tf.keras.models.Sequential()
            self.q_model.add(tf.keras.Input(shape=(self.SHAPE,), batch_size=self.N_SAMPLES))
            self.q_model.add(tf.keras.layers.Dense(64, activation='relu'))
            self.q_model.add(tf.keras.layers.Dense(128, activation='relu'))
            self.q_model.add(tf.keras.layers.Dense(self.N_ACTIONS))
            self.q_model.build((None,))
            self.q_model.summary()
            self.q_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.ALPHA, momentum=0.95),
                                 loss=tf.keras.losses.MeanSquaredError(),
                                 metrics=['accuracy'], run_eagerly=False)

        else:
            self.q_model = tf.keras.models.load_model(self.model_path)
        
        # Make Q-target
        self.q_target = tf.keras.models.clone_model(self.q_model)

    def train_q(self, replay_buffer):
        # replay_buffer: state tuples ([state, action, reward, state_next, done])
        batch_size = len(replay_buffer)
        subsample = np.random.choice(batch_size, size=self.N_SAMPLES, replace=False)

        train_set = np.array([np.array(replay_buffer[step][0]) for step in subsample])
        action_set = np.array([replay_buffer[step][1] for step in subsample])
        reward_list = np.array([replay_buffer[step][2] for step in subsample])
        next_states = np.array([np.array(replay_buffer[step][3]) for step in subsample])
        done_list = np.array([replay_buffer[step][4] for step in subsample])

        target_set = self.q_predict(train_set)
        target_scalar = self.target_value(next_states)

        for n, (done, action, reward, step) in enumerate(zip(done_list, action_set, reward_list, target_scalar)):
            if done:
                target_value = reward
            else:
                target_value = reward + (self.GAMMA * step)

            target_set[n][action] = target_value

        # Fit predictions to target
        # history = self.q_model.fit(x=train_set, y=target_set, epochs=self.EPOCHS)
        # accuracy = history.history['accuracy']
        # loss = history.history['loss']
        self.q_model.fit(x=train_set, y=target_set, epochs=1)
        return None

    def q_predict_action(self, state):
        # Expecting state with shape = [[...], [...], ...]
        predictions = self.q_model.predict(state)
        greedy_action = tf.math.argmax(predictions, axis=1)
        return greedy_action

    def q_predict(self, state):
        # Expecting state with shape = [[...], [...], ...]
        predictions = self.q_model.predict(state)
        return predictions

    def target_value(self, state):
        predictions = self.q_target.predict(state)
        value = tf.math.reduce_max(predictions, axis=1)
        return value

    def avg_q_value(self, state):
        predictions = self.q_model.predict(state)
        avg_per_action = predictions.mean(0)
        avg_per_state = tf.math.reduce_mean(predictions, axis=1).numpy()
        avg_q_scalar = tf.math.reduce_mean(avg_per_state).numpy()
        return avg_per_action, avg_per_state, avg_q_scalar

    def copy_model(self):
        # self.q_target = tf.keras.models.clone_model(self.q_model)
        self.q_target.set_weights(self.q_model.get_weights())
        return None

    def save_model(self):
        self.q_model.save('./q_model.h5')
        return None
