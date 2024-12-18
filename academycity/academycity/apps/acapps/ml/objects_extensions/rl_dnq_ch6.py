from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from ....core.utils import log_debug, clear_log_debug
#
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras import Model
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.signal import convolve #, gaussian
from scipy.stats import norm
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from base64 import b64encode

from IPython.display import HTML
#
from tqdm import trange
from IPython.display import clear_output

seed = 13
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

class ReplayBuffer1:
    def __init__(self, size):
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to holde buffer
        self.next_id = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition):
        # transition = (state, action, reward, next_state, done)
        # item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(transition)
        else:
            self.buffer[self.next_id] = transition
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size, state_size):
        # idxs = np.random.choice(len(self.buffer), batch_size)
        # samples = [self.buffer[i] for i in idxs]
        # states, actions, rewards, next_states, done_flags = list(zip(*samples))
        # return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)

        minibatch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state = np.zeros((batch_size, state_size))
        next_state = np.zeros((batch_size, state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        return state, action, reward, next_state,done

    def is_ready(self):
        if len(self.buffer) >= self.size:
            return True
        else:
            return False


class ReplayBuffer:
    def __init__(self, size):
        self.size = size  # max number of items in buffer
        self.buffer = []  # array to holde buffer
        self.next_id = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)

    def is_full(self):
        return len(self.buffer) >= self.size


class DQNAgent:
    def __init__(self, env, replay_size = 10**4, epsilon=0.0, gamma = 0.99):
        self.epsilon = epsilon
        self.gamma = gamma
        #
        self.start_epsilon = 1
        self.end_epsilon = 0.05
        self.eps_decay_final_step = 2 * 10 ** 4
        #
        self.env = env
        self.state_shape, self.n_actions = self.env.observation_space.shape, self.env.action_space.n
        print(f"state shape:{self.state_shape}\nNumber of Actions:{self.n_actions}")
        #
        self.exp_replay = ReplayBuffer(size = replay_size)
        #
        self.model = None
        self.target_model = None
        self.create_model()
        #
        state, _ = env.reset(seed=123)
        plt.imshow(self.env.render())
        n = 0
        for i in range(100):
            n += 1
            self.play_and_record(state, n_steps=10 ** 2)
            if n > replay_size:
                break
        print(len(self.exp_replay.buffer))

    def __call__(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        q_values = self.model(state_t)
        return q_values

    def create_model(self):
        state_dim = self.state_shape[0]
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(state_dim,)))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.n_actions))
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        q_values = self.model(states)
        return q_values.numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

    def evaluate(self, n_games=1, greedy=False, t_max=10000):
        rewards = []
        for _ in range(n_games):
            s, _ = self.env.reset()
            reward = 0
            for _ in range(t_max):
                qvalues = self.get_qvalues(np.array([s]))
                action = qvalues.argmax(axis=-1)[0] if greedy else self.sample_actions(qvalues)[0]
                s, r, terminated, truncated, _ = self.env.step(action)
                reward += r
                if terminated:
                    break

            rewards.append(reward)
        return np.mean(rewards)

    def play_and_record(self, start_state, n_steps=1):
        s = start_state
        sum_rewards = 0

        # Play the game for n_steps and record transitions in buffer
        for _ in range(n_steps):
            qvalues = self.get_qvalues(np.array([s]))
            a = self.sample_actions(qvalues)[0]
            next_s, r, terminated, truncated, _ = self.env.step(a)
            sum_rewards += r
            done = terminated or truncated
            self.exp_replay.add(s, a, r, next_s, done)
            if terminated:
                s, _ = self.env.reset()
            else:
                s = next_s

        return sum_rewards, s

    def compute_td_loss(self, states, actions, rewards, next_states, done_flags):
        # get q-values for all actions in current states
        # use agent network
        predicted_qvalues = self.model(states)
        predicted_next_qvalues = self.target_model(next_states)

        # select q-values for chosen actions
        row_indices = tf.range(len(actions))
        indices = tf.transpose([row_indices, actions])
        predicted_qvalues_for_actions = tf.gather_nd(predicted_qvalues, indices)

        # compute Qmax(next_states, actions) using predicted next q-values
        next_state_values = tf.reduce_max(predicted_next_qvalues, axis=1)

        # compute "target q-values"
        target_qvalues_for_actions = rewards + self.gamma * next_state_values * (1 - done_flags)

        # mean squared error loss to minimize
        loss = tf.keras.losses.MSE(target_qvalues_for_actions, predicted_qvalues_for_actions)

        return loss

    def epsilon_schedule(self, step):
        return (self.start_epsilon + (self.end_epsilon - self.start_epsilon) *
                min(step, self.eps_decay_final_step) / self.eps_decay_final_step)

    def smoothen(self, values):
        # Generate a symmetric range centered around 0 (e.g., from -50 to 50)
        x = np.linspace(-100 // 2, 100 // 2, 100)
        kernel = norm.pdf(x, loc=0, scale=100)
        kernel /= kernel.sum()
        return convolve(values, kernel, 'valid')

    def train(self):
        # setup some parameters for training
        timesteps_per_epoch = 1
        batch_size = 32
        total_steps = 5 * 10 ** 4

        # setup spme frequency for loggind and updating target network
        loss_freq = 20
        refresh_target_network_freq = 100
        eval_freq = 1000

        # to clip the gradients
        max_grad_norm = 5000

        # init Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # ===
        state, _ = self.env.reset()
        for step in trange(total_steps + 1):
            # reduce exploration as we progress
            self.epsilon = self.epsilon_schedule(step)

            # take timesteps_per_epoch and update experience replay buffer
            _, state = self.play_and_record(state, timesteps_per_epoch)

            # train by sampling batch_size of data from experience replay
            states, actions, rewards, next_states, done_flags = self.exp_replay.sample(batch_size)

            with tf.GradientTape() as tape:
                # loss = <compute TD loss>
                loss = self.compute_td_loss(states, actions, rewards, next_states, done_flags)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            clipped_grads = [tf.clip_by_norm(g, max_grad_norm) for g in gradients]
            optimizer.apply_gradients(zip(clipped_grads, agent.model.trainable_variables))

            if step % loss_freq == 0:
                td_loss_history.append(loss.numpy())

            if step % refresh_target_network_freq == 0:
                # Load agent weights into target_network
                target_network.model.set_weights(agent.model.get_weights())

            if step % eval_freq == 0:
                # eval the agent
                mean_rw_history.append(evaluate(
                    make_env(env_name), agent, n_games=3, greedy=True, t_max=1000)
                )

                clear_output(True)
                print("buffer size = %i, epsilon = %.5f" %
                      (len(exp_replay), agent.epsilon))

                plt.figure(figsize=[16, 5])
                plt.subplot(1, 2, 1)
                plt.title("Mean return per episode")
                plt.plot(mean_rw_history)
                plt.grid()

                assert not np.isnan(td_loss_history[-1])
                plt.subplot(1, 2, 2)
                plt.title("TD loss history (smoothened)")
                plt.plot(smoothen(td_loss_history))
                plt.grid()

                plt.show()



class DQNAgent1:
    def __init__(self):
        self.env_id = 'CartPole-v1'


        self.replay_buffer = ReplayBuffer(size = 2000)
        # create main model
        self.model = self.create_model()

    def create_model(self):
        X_input = Input((self.state_size,))
        action_space = self.action_size

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 51  nodes
        X = Dense(512, input_shape=(self.state_size,), activation="relu", kernel_initializer='he_uniform')(X_input)

        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Output Layer with # of actions:   nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='CartPole')
        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        model.summary()
        return model

    # def update_replay_memory(self, transition):
    #     self.memory.append(transition)
    #     if len(self.memory) > self.train_start:
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):

        # A33
        # # Randomly sample minibatch from the memory
        # minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        # state = np.zeros((self.batch_size, self.state_size))
        # next_state = np.zeros((self.batch_size, self.state_size))
        # action, reward, done = [], [], []
        #
        # # do this before prediction
        # # for speedup, this could be done on the tensor level
        # # but easier to understand using a loop
        # for i in range(self.batch_size):
        #     state[i] = minibatch[i][0]
        #     action.append(minibatch[i][1])
        #     reward.append(minibatch[i][2])
        #     next_state[i] = minibatch[i][3]
        #     done.append(minibatch[i][4])

        # do batch prediction to save speed
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.state_size)

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def train(self, file_name, episodes):
        clear_log_debug()
        for e in range(episodes):
            log_debug("Episode A: " + str(e))
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            log_debug("Episode AA: " + str(e))
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100

                # A33
                # self.update_replay_memory((state, action, reward, next_state, done))
                self.replay_buffer.add((state, action, reward, next_state, done))
                if self.replay_buffer.is_ready():
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                    self.replay()

                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.keras")
                        log_debug("file_name: " + file_name)
                        log_debug("Episode Z1: ")
                        self.save(file_name)
                        log_debug("Episode Z2: ")


            log_debug("Episode B: " + str(e))
        log_debug("End Train...")

    def test(self, file_name, episodes):
        clear_log_debug()
        self.load(file_name)
        all_states = []
        for e in range(episodes):
            log_debug("Episode A: " + str(e))
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            episode_states = []
            for time in range(500):
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                episode_states.append(next_state.tolist())
                state = np.reshape(next_state, [1, self.state_size])
                if done:
                    break
            all_states.append(episode_states)
            log_debug("Episode B: " + str(e))
        log_debug("End Test...")
        return all_states

    def record_video(self, video_folder, video_length):

        # vec_env = DummyVecEnv([lambda: gym.make(self.env_id, render_mode="rgb_array")])
        # # Record the video starting at the first step
        # vec_env = VecVideoRecorder(vec_env, video_folder,
        #                        record_video_trigger=lambda x: x == 0, video_length=video_length,
        #                        name_prefix=f"{type(self).__name__}-{self.env_id}")

        # Initialize DummyVecEnv without render_mode
        vec_env = DummyVecEnv([lambda: gym.make(self.env_id)])

        # Set up VecVideoRecorder to record the video
        vec_env = VecVideoRecorder(vec_env, video_folder,
            record_video_trigger=lambda x: x == 0, video_length=video_length,
            name_prefix=f"{type(self).__name__}-{self.env_id}"
        )

        obs = vec_env.reset()
        for _ in range(video_length + 1):
            # action = np.argmax(self.get_qvalues(obs),axis=-1)
            action = np.argmax(self.model.predict(obs))
            obs, _, _, _ = vec_env.step(action)
        # video filename
        file_path = "./"+video_folder+vec_env.video_recorder.path.split("/")[-1]
        # Save the video
        vec_env.close()
        return file_path

    def play_video(self, file_path):
        mp4 = open(file_path, 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        return HTML("""
            <video width=400 controls>
                  <source src="%s" type="video/mp4">
            </video>
            """ % data_url)



class DQNAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 DNQAlgo\n", dic, '\n', '-'*50)
        try:
            super(DQNAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 DNQAlgo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 DNQAlgo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class DQNDataProcessing(BaseDataProcessing, BasePotentialAlgo, DQNAlgo):
    def __init__(self, dic):
        # print("90567-010 DNQDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DNQDataProcessing ", self.app)

        self.env_name = 'CartPole-v1'
        self.env = gym.make(self.env_name, render_mode="rgb_array", max_episode_steps=4000)

    def train(self, dic):
        print("90155-dqn: \n", "="*50, "\n", dic, "\n", "="*50)

        # n_timesteps = 60  # Lookback window of 60 days
        # n_features = 5  # e.g., OHLC and volume
        #
        # # Example of running the model (for simplicity, using random data)
        # X = np.random.rand(1000, n_timesteps, n_features)  # Simulated stock data
        # y = np.random.randint(0, 3, size=(1000,))  # Random actions (buy/sell/hold)
        # print(X, "\n\n", X.shape, "\n\n", y)
        #
        #
        # return

        episodes = int(dic["episodes"])
        # ---
        agent = DQNAgent(self.env, replay_size = 10**4, epsilon = 0.5, gamma = 0.99)
        save_to_file = os.path.join(self.TO_OTHER, "cartpole-dqn.keras")

        print(save_to_file)

        # agent.train(save_to_file, episodes)

        result = {"status": "ok dqn"}
        return result

    def test(self, dic):
        print("90155-dqn: \n", "="*50, "\n", dic, "\n", "="*50)
        episodes = int(dic["episodes"])
        # ---

        agent = DQNAgent()
        read_from_file = os.path.join(self.TO_OTHER, "cartpole-dqn.keras")
        results = agent.test(read_from_file, episodes)
        for i in results:
            print(i)

        result = {"status": "ok dqn", "results": results}
        return result

    def record_video(self, dic):
        print("9034-dqn-video: \n", "="*50, "\n", dic, "\n", "="*50)

        video_folder = self.TO_OTHER
        video_length = 500
        agent = DQNAgent()
        agent.record_video(video_folder, video_length)



        result = {"status": "ok dqn record_video"}
        return result

