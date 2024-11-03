from tensorflow.python.keras.backend_config import epsilon

from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo

from ....core.utils import log_debug, clear_log_debug

from stable_baselines3 import DQN

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ======================================================================

from random import randint, choice
from time import sleep
import pygame, time
from keras.models import Sequential, Model, load_model

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers import Input, BatchNormalization, GlobalMaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
# import keras.backend.tensorflow_backend as backend
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import random
import os

import numpy as np
import pandas as pd
from collections import deque
import pickle

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

pygame.init()
TstartTime = time.time()


######################################################################################
class Field:
    def __init__(self, height=10, width=5):
        self.width = width
        self.height = height
        self.body = np.zeros(shape=(self.height, self.width))
        self.field_records = []

    def update_field(self, walls, player):
        try:
            # Clear the field:
            self.body = np.zeros(shape=(self.height, self.width))
            # Put the walls on the field:
            for wall in walls:
                if not wall.out_of_range:
                    self.body[wall.y:min(wall.y + wall.height, self.height), :] = wall.body

            # Put the player on the field:
            self.body[player.y:player.y + player.height, player.x:player.x + player.width] += player.body
            self.field_records.append(
                [[player.x, player.width],
                 [walls[-1].y, walls[-1].hole_pos, walls[-1].hole_width]]
            )
        except:
            pass


######################################################################################
class Wall:
    def __init__(self, height=5, width=100, hole_width=20,
                 y=0, speed=1, field=None):
        self.height = height
        self.width = width
        self.hole_width = hole_width
        self.y = y
        self.speed = speed
        self.field = field
        self.body_unit = 1
        self.body = np.ones(shape=(self.height, self.width)) * self.body_unit
        self.out_of_range = False
        self.hole_pos = 0
        self.create_hole()

    def create_hole(self):
        # hole = np.zeros(shape=(self.height, self.hole_width))
        self.hole_pos = randint(0, self.width - self.hole_width)
        self.body[:, self.hole_pos:self.hole_pos + self.hole_width] = 0

    def move(self):
        self.y += self.speed
        self.out_of_range = True if ((self.y + self.height) > self.field.height) else False


######################################################################################
class Player:
    def __init__(self, height=5, max_width=10, width=2,
                 x=0, y=0, speed=2):
        self.height = height
        self.max_width = max_width
        self.width = width
        self.x = x
        self.y = y
        self.speed = speed
        self.body_unit = 2
        self.body = np.ones(shape=(self.height, self.width)) * self.body_unit
        self.stamina = 20
        self.max_stamina = 20

    def move(self, field, direction=0):
        '''
        Moves the player :
         - No change          = 0
         - left, if direction  = 1
         - right, if direction = 2
        '''
        val2dir = {0: 0, 1: -1, 2: 1}
        direction = val2dir[direction]
        next_x = (self.x + self.speed * direction)
        if not (next_x + self.width > field.width or next_x < 0):
            self.x += self.speed * direction
            self.stamina -= 1

    def change_width(self, action=0):
        '''
        Change the player's width:
         - No change          = 0
         - narrow by one unit = 3
         - widen by one unit  = 4
        '''
        val2act = {0: 0, 3: -1, 4: 1}
        action = val2act[action]
        new_width = self.width + action
        player_end = self.x + new_width
        if new_width <= self.max_width and new_width > 0 and player_end <= self.max_width:
            self.width = new_width
            self.body = np.ones(shape=(self.height, self.width)) * self.body_unit


######################################################################################
class Environment:
    P_HEIGHT = 2   # Height of the player
    F_HEIGHT = 20  # Height of the field
    W_HEIGHT = 2   # Height of the walls
    WIDTH = 10  # Width of the field and the walls
    MIN_H_WIDTH = 2   # Minimum width of the holes
    MAX_H_WIDTH = 6  # Maximum width of the holes
    MIN_P_WIDTH = 2   # Minimum Width of the player
    MAX_P_WIDTH = 6  # Maximum Width of the player
    HEIGHT_MUL = 30  # Height Multiplier (used to draw np.array as blocks in pygame )
    WIDTH_MUL = 40  # Width Multiplier (used to draw np.array as blocks in pygame )
    WINDOW_HEIGHT = (F_HEIGHT + 1) * HEIGHT_MUL  # Height of the pygame window
    WINDOW_WIDTH = (WIDTH) * WIDTH_MUL  # Widh of the pygame window

    ENVIRONMENT_SHAPE = (F_HEIGHT, WIDTH, 1)
    ACTION_SPACE = [0, 1, 2, 3, 4]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)
    PUNISHMENT = -100  # Punishment increment
    REWARD = 10  # Reward increment
    score = 0  # Initial Score

    MOVE_WALL_EVERY = 4  # Every how many frames the wall moves.
    MOVE_PLAYER_EVERY = 1  # Every how many frames the player moves.
    frames_counter = 0
    score_increased = False

    def __init__(self):
        # Colors:
        self.BLACK = (25, 25, 25)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 80, 80)
        self.BLUE = (80, 80, 255)
        self.field = self.walls = self.player = None
        self.current_state = self.reset()
        self.MAX_VAL = 3
        self.val2color = {0: self.WHITE, self.walls[0].body_unit: self.BLACK,
                          self.player.body_unit: self.BLACK, self.MAX_VAL: self.RED}
        self.game_over = False
        self.episodes_records = {}

    def reset(self):
        self.score = 0
        self.frames_counter = 0
        self.game_over = False

        self.field = Field(height=self.F_HEIGHT, width=self.WIDTH)
        w1 = Wall(height=self.W_HEIGHT, width=self.WIDTH,
                  hole_width=randint(self.MIN_H_WIDTH, self.MAX_H_WIDTH),
                  field=self.field)
        self.walls = deque([w1])
        p_width = randint(self.MIN_P_WIDTH, self.MAX_P_WIDTH)
        self.player = Player(height=self.P_HEIGHT, max_width=self.WIDTH, width=p_width,
                             x=randint(0, self.field.width - p_width),
                             y=int(self.field.height * 0.7), speed=1)
        self.MAX_VAL = self.player.body_unit + w1.body_unit
        # Update the field :
        self.field.update_field(self.walls, self.player)

        observation = self.field.body / self.MAX_VAL
        return observation

    def print_text(self, WINDOW=None, text_cords=(0, 0), center=False,
                   text="", color=(0, 0, 0), size=32):
        pygame.init()
        font = pygame.font.Font('freesansbold.ttf', size)
        text_to_print = font.render(text, True, color)
        textRect = text_to_print.get_rect()
        if center:
            textRect.center = text_cords
        else:
            textRect.x = text_cords[0]
            textRect.y = text_cords[1]
        WINDOW.blit(text_to_print, textRect)

    def step(self, action):

        self.frames_counter += 1
        reward = 0

        # If the performed action is (move) then player.move method is called:
        if action in [1, 2]:
            self.player.move(direction=action, field=self.field)
        # If the performed action is (change_width) then player.change_width method is called:
        if action in [3, 4]:
            self.player.change_width(action=action)

            # Move the wall one step (one step every MOVE_WALL_EVERY frames):
        if self.frames_counter % self.MOVE_WALL_EVERY == 0:
            # move the wall one step
            self.walls[-1].move()
            # reset the frames counter
            self.frames_counter = 0

        # Update the field :
        self.field.update_field(self.walls, self.player)

        # If the player passed a wall successfully increase the reward +1
        if (self.walls[-1].y == (self.player.y + self.player.height)) and not self.score_increased:
            reward += self.REWARD
            self.score += self.REWARD

            # Increase player's stamina every time it passed a wall successfully
            self.player.stamina = min(self.player.max_stamina, self.player.stamina + 10)
            # self.score_increased : a flag to make sure that reward increases once per wall
            self.score_increased = True

        #  Lose Conditions :
        # C1 : The player hits a wall
        # C  : Player's width was far thinner than hole's width
        # C3 : Player fully consumed its stamina (energy)
        lose_conds = [self.MAX_VAL in self.field.body,
                      ((self.player.y == self.walls[-1].y) and (self.player.width < (self.walls[-1].hole_width - 1))),
                      self.player.stamina <= 0]

        # If one lose condition or more happend, the game ends:
        if True in lose_conds:
            self.game_over = True
            reward = self.PUNISHMENT
            return self.field.body / self.MAX_VAL, reward, self.game_over

        # Check if a wall moved out of the scene:
        if self.walls[-1].out_of_range:
            # Create a new wall
            self.walls[-1] = Wall(height=self.W_HEIGHT, width=self.WIDTH,
                                  hole_width=randint(self.MIN_H_WIDTH, self.MAX_H_WIDTH),
                                  field=self.field)

            self.score_increased = False

        # Return New Observation , reward, game_over(bool)
        return self.field.body / self.MAX_VAL, reward, self.game_over

    def render(self, WINDOW=None, human=False):
        if human:
            ################ Check Actions #####################
            action = 0
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 1
                    if event.key == pygame.K_RIGHT:
                        action = 2

                    if event.key == pygame.K_UP:
                        action = 4
                    if event.key == pygame.K_DOWN:
                        action = 3
            ################## Step ############################
            _, reward, self.game_over = self.step(action)
        ################ Draw Environment ###################
        WINDOW.fill(self.WHITE)
        self.field.update_field(self.walls, self.player)
        for r in range(self.field.body.shape[0]):
            for c in range(self.field.body.shape[1]):
                pygame.draw.rect(WINDOW,
                                 self.val2color[self.field.body[r][c]],
                                 (c * self.WIDTH_MUL, r * self.HEIGHT_MUL, self.WIDTH_MUL, self.HEIGHT_MUL))

        self.print_text(WINDOW=WINDOW, text_cords=(self.WINDOW_WIDTH // 2, int(self.WINDOW_HEIGHT * 0.1)),
                        text=str(self.score), color=self.RED, center=True)
        self.print_text(WINDOW=WINDOW, text_cords=(0, int(self.WINDOW_HEIGHT * 0.9)),
                        text=str(self.player.stamina), color=self.RED)

        pygame.display.update()

    def record_episode(self, episode):
        self.episodes_records[episode] = self.field.field_records
        self.field.field_records = []

    def save_recorded_episodes(self, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(self.episodes_records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_recorded_episodes(self, file_name):
        with open(file_name, 'rb') as handle:
            results = pickle.load(handle)
        return results


######################################################################################
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    def set_model(self, model):
        self.model = model
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter
        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()


######################################################################################
class DQNAgent:
    def __init__(self, name, env, conv_list, dense_list, util_list, path, MINIBATCH_SIZE):
        self.path = path
        self.env = env
        self.conv_list = conv_list
        self.dense_list = dense_list
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.name = [str(name) + " | " + "".join(str(c) + "C | " for c in conv_list) + "".join(
            str(d) + "D | " for d in dense_list) + "".join(u + " | " for u in util_list)][0]
        self.model = None
        self.target_model = None
        self.epsilon = None
        # ---
        self.load_model()
        # ---

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(name, log_dir="{}logs/{}-{}".format(path, name, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Creates a convolutional block given (filters) number of filters, (dropout) dropout rate,
    # (bn) a boolean variable indecating the use of BatchNormalization,
    # (pool) a boolean variable indecating the use of MaxPooling2D
    def conv_block(self, inp, filters=64, bn=True, pool=True, dropout=0.2):
        _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPooling2D(pool_size=(2, 2))(_)
        if dropout > 0:
            _ = Dropout(0.2)(_)
        return _

    # Creates the model with the given specifications:
    def create_model(self):
        conv_list = self.conv_list
        dense_list = self.dense_list
        # Defines the input layer with shape = ENVIRONMENT_SHAPE
        input_layer = Input(shape=self.env.ENVIRONMENT_SHAPE)
        # Defines the first convolutional block:
        _ = self.conv_block(input_layer, filters=conv_list[0], bn=False, pool=False)
        # If number of convolutional layers is   or more, use a loop to create them.
        if len(conv_list) > 1:
            for c in conv_list[1:]:
                _ = self.conv_block(_, filters=c)
        # Flatten the output of the last convolutional layer.
        _ = Flatten()(_)

        # Creating the dense layers:
        for d in dense_list:
            _ = Dense(units=d, activation='relu')(_)
        # The output layer has 5 nodes (one node per action)
        output = Dense(units=self.env.ACTION_SPACE_SIZE,
                       activation='linear', name='output')(_)

        # Put it all together:
        model = Model(inputs=input_layer, outputs=[output])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss={'output': 'mse'},
                      metrics={'output': 'accuracy'})
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def act(self, state, epsilon):
        if np.random.random() <= epsilon:
            return choice(self.env.ACTION_SPACE)
        else:
            return np.argmax(self.model.predict(state.reshape(-1, *self.env.ENVIRONMENT_SHAPE)))

    def replay(self, game_over):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states.reshape(-1, *self.env.ENVIRONMENT_SHAPE))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states.reshape(-1, *self.env.ENVIRONMENT_SHAPE))

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # self.model.fit(x=np.array(X).reshape(-1, *self.env.ENVIRONMENT_SHAPE),
        #                y=np.array(y),
        #                batch_size=self.MINIBATCH_SIZE, verbose=0,
        #                shuffle=False, callbacks=[] if game_over else None)

        self.model.fit(x=np.array(X).reshape(-1, *self.env.ENVIRONMENT_SHAPE),
                       y=np.array(y),
                       batch_size=self.MINIBATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensorboard] if game_over else None)

        # Update target network counter every episode
        if game_over:
            self.target_update_counter += 1

    def load_model(self):
        checkpoint_name = f"{self.name}.model"
        file_name = f'{self.path}models/{checkpoint_name}'
        # Main model
        if os.path.exists(file_name):
            print(f"Loading model from {file_name}")
            self.model = load_model(file_name)
        else:
            print("Creating new model")
            self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        # ---
        epsilon_name = f"{self.name}.pkl"
        epsilon_file_name = f'{self.path}pickles/{epsilon_name}'
        if os.path.exists(epsilon_file_name):
            print(f"Loading epsilon from {epsilon_file_name}")
            with open(epsilon_file_name, 'rb') as handle:
                self.epsilon = pickle.load(handle)

    def save_model(self):
        checkpoint_name = f"{self.name}.model"
        file_name = f'{self.path}models/{checkpoint_name}'
        self.model.save(file_name)

        epsilon_name = f"{self.name}.pkl"
        epsilon_file_name = f'{self.path}pickles/{epsilon_name}'
        with open(epsilon_file_name, 'wb') as handle:
            pickle.dump(self.epsilon, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, episode, max_reward, average_reward, min_reward, path):
        checkpoint_name = f"{self.name}| Eps({episode}) | max({max_reward:_>7.2f}) | avg({average_reward:_>7.2f}) | min({min_reward:_>7.2f}).model"
        self.model.save(f'{path}models/{checkpoint_name}')
        best_weights = self.model.get_weights()
        print("save: ", path)
        return best_weights

    # Trains main network every step during episode
    def train(self, m, epsilon, best_average, ecc_enabled, epsilon_decay, max_epsilon, ef_enabled, avg_reward_info,
              fluctuate_every, EPISODES):

        if self.epsilon is None:
            self.epsilon = epsilon

        if ecc_enabled: MAX_EPS_NO_INC = m["ECC_Settings"]["MAX_EPS_NO_INC"]
        MIN_EPSILON = 0.001  # Minimum epsilon value
        EPSILON_DECAY = epsilon_decay
        MAX_EPSILON = max_epsilon
        EF_Enabled = ef_enabled
        FLUCTUATE_EVERY = fluctuate_every
        best_score = -100
        max_reward_info = [[1, best_score, self.epsilon]]
        eps_no_inc_counter = 0  # Counts episodes with no increment in reward
        best_weights = [self.model.get_weights()]
        ep_rewards = [best_average]

        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            print("Episode: ", episode, "\n", "="*30)
            if m["best_only"]: self.model.set_weights(best_weights[0])
            self.env.score_increased = False
            # Update tensorboard step every episode
            self.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            # ==
            step = 1
            # action = 0
            current_state = self.env.reset()
            # print(current_state)

            game_over = self.env.game_over
            while not game_over:
                # This part stays mostly the same, the change is to query a model for Q values
                action = self.act(current_state, self.epsilon)
                new_state, reward, game_over = self.env.step(action)
                # Transform new continuous state to new discrete state and count reward
                episode_reward += reward

                # Uncomment the next block if you want to show preview on your screen
                # if SHOW_PREVIEW and not episode % SHOW_EVERY:
                #     clock.tick(27)
                #     env.render(WINDOW)

                # Every step we update replay memory and train main network
                self.update_replay_memory((current_state, action, reward, new_state, game_over))
                self.replay(game_over)
                # If counter reaches set value, update target network with weights of main network
                if self.target_update_counter > UPDATE_TARGET_EVERY:
                    self.target_model.set_weights(self.model.get_weights())
                    self.target_update_counter = 0

                current_state = new_state
                step += 1
            # ==
            self.env.record_episode(episode)
            # ==
            if ecc_enabled: eps_no_inc_counter += 1
            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)

            if not episode % AGGREGATE_STATS_EVERY:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                self.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                               reward_max=max_reward,
                                               epsilon=self.epsilon)

                # Save models, but only when avg reward is greater or equal a set value
                if not episode % SAVE_MODEL_EVERY:
                    _ = self.save(episode, max_reward, average_reward, min_reward, self.path)

                if average_reward > best_average:
                    best_average = average_reward
                    # update ECC variables:
                    avg_reward_info.append([episode, best_average, self.epsilon])
                    eps_no_inc_counter = 0
                    best_weights[0] = self.save(episode, max_reward, average_reward,
                                                 min_reward, self.path)
                    self.save_model()

                if ecc_enabled and eps_no_inc_counter >= MAX_EPS_NO_INC:
                    self.epsilon = avg_reward_info[-1][2]  # Get epsilon value of the last best reward
                    eps_no_inc_counter = 0

            if episode_reward > best_score:
                try:
                    best_score = episode_reward
                    max_reward_info.append([episode, best_score, self.epsilon])

                    # Save Agent :
                    best_weights[0] = self.save(episode, max_reward, average_reward,
                                                min_reward, self.path)
                    self.save_model()
                except:
                    pass

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                self.epsilon *= EPSILON_DECAY
                self.epsilon = max(MIN_EPSILON, self.epsilon)

            # Epsilon Fluctuation:
            if EF_Enabled:
                if not episode % FLUCTUATE_EVERY:
                    self.epsilon = MAX_EPSILON
        return round(sum(ep_rewards) / len(ep_rewards), 2), max_reward_info, avg_reward_info

######################################################################################

######################################################################################
# ## Constants:
# RL Constants:
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 3000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)
MIN_REWARD = 1000  # For model save
SAVE_MODEL_EVERY = 1000  # Episodes
SHOW_EVERY = 20  # Episodes
#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW = False
######################################################################################
# Models Arch :
# [{[conv_list], [dense_list], [util_list], MINIBATCH_SIZE, {EF_Settings}, {ECC_Settings}} ]

######################################################################################





# https://github.com/ModMaamari/reinforcement-learning-using-python

class DNQCNNAlgo(object):
    def __init__(self, dic):
        # print("90567-8-000 DNQCNNAlgo\n", dic, '\n', '-'*50)
        try:
            super(DNQCNNAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 DNQCNNAlgo:\n"+str(ex), "\n", '-'*50)
        # print("DNQCNNAlgo\n", self.app)
        # print("90004-020 DNQCNNAlgo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class DNQCNNDataProcessing(BaseDataProcessing, BasePotentialAlgo, DNQCNNAlgo):
    def __init__(self, dic):
        # print("90567-010 DNQCNNDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DNQCNNDataProcessing ", self.app)

        self.PATH = os.path.join(self.TO_OTHER, "dqncnn")
        print(f'{self.PATH}models')

        if not os.path.isdir(f'{self.PATH}models'):
            os.makedirs(f'{self.PATH}models')

        if not os.path.isdir(f'{self.PATH}results'):
            os.makedirs(f'{self.PATH}results')

        if not os.path.isdir(f'{self.PATH}pickles'):
            os.makedirs(f'{self.PATH}pickles')

        self.MINIBATCH_SIZE = None

    def run(self, dic):
        print("90300-50: \n", "="*50, "\n", dic, "\n", "="*50)
        clear_log_debug()
        log_debug("dqn-cnn run")
        EPISODES = int(dic["episodes"])

        models_arch = [{"conv_list": [32], "dense_list": [32, 32], "util_list": ["ECC2", "1A-5Ac"],
                        "MINIBATCH_SIZE": 128, "best_only": False,
                        "EF_Settings": {"EF_Enabled": False}, "ECC_Settings": {"ECC_Enabled": False}},

                       {"conv_list": [32], "dense_list": [32, 32, 32], "util_list": ["ECC2", "1A-5Ac"],
                        "MINIBATCH_SIZE": 128, "best_only": False,
                        "EF_Settings": {"EF_Enabled": False}, "ECC_Settings": {"ECC_Enabled": False}},

                       {"conv_list": [32], "dense_list": [32, 32], "util_list": ["ECC2", "1A-5Ac"],
                        "MINIBATCH_SIZE": 128, "best_only": False,
                        "EF_Settings": {"EF_Enabled": True, "FLUCTUATIONS": 2},
                        "ECC_Settings": {"ECC_Enabled": True, "MAX_EPS_NO_INC": int(EPISODES * 0.2)}}]

        # A dataframe used to store grid search results
        res = pd.DataFrame(columns=["Model Name", "Convolution Layers", "Dense Layers", "Batch Size", "ECC", "EF",
                                    "Best Only", "Average Reward", "Best Average", "Epsilon 4 Best Average",
                                    "Best Average On", "Max Reward", "Epsilon 4 Max Reward", "Max Reward On",
                                    "Total Training Time (min)", "Time Per Episode (sec)"])

        # Grid Search:
        for i, m in enumerate(models_arch):
            # if i == 1 or i == 2:
            #     continue

            log_debug("dqn-cnn run: " + str(i))

            startTime = time.time()  # Used to count episode training time
            self.MINIBATCH_SIZE = m["MINIBATCH_SIZE"]

            # Exploration settings :
            # Epsilon Fluctuation (EF):
            EF_Enabled = m["EF_Settings"]["EF_Enabled"]  # Enable Epsilon Fluctuation
            MAX_EPSILON = 1  # Maximum epsilon value

            FLUCTUATE_EVERY = None
            if EF_Enabled:
                FLUCTUATIONS = m["EF_Settings"]["FLUCTUATIONS"]  # How many times epsilon will fluctuate
                FLUCTUATE_EVERY = int(EPISODES / FLUCTUATIONS)  # Episodes
                EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / FLUCTUATE_EVERY)
                epsilon = 1  # not a constant, going to be decayed
            else:
                EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / (0.8 * EPISODES))
                epsilon = 1  # not a constant, going to be decayed

            # Initialize some variables:
            best_average = -100
            # Epsilon Conditional Constantation (ECC):
            log_debug("dqn-cnn run: B " + str(i))
            ECC_Enabled = m["ECC_Settings"]["ECC_Enabled"]
            avg_reward_info = [
                [1, best_average, epsilon]]  # [[episode1, reward1 , epsilon1] ... [episode_n, reward_n , epsilon_n]]

            # Maximum number of episodes without any increment in reward average
            # -------------------
            env = Environment()
            env.MOVE_WALL_EVERY = 1  # Every how many frames the wall moves.

            log_debug("dqn-cnn run: C " + str(i))

            agent = DQNAgent(f"M{i}", env, m["conv_list"], m["dense_list"], m["util_list"], self.PATH, self.MINIBATCH_SIZE)
            MODEL_NAME = agent.name
            # -------------------

            # Uncomment these two lines if you want to show preview on your screen
            # WINDOW          = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
            # clock           = pygame.time.Clock()


            # ===============================================
            # Get Average reward:

            log_debug("dqn-cnn run: D " + str(i))

            average_reward, max_reward_info, avg_reward_info = agent.train(m, epsilon, best_average, ECC_Enabled,
                                                                           EPSILON_DECAY, MAX_EPSILON, EF_Enabled,
                                                                           avg_reward_info, FLUCTUATE_EVERY, EPISODES)

            file_name = f'{self.PATH}pickles/{"result_for_M"+ str(i+1) +".pkl"}'

            log_debug("dqn-cnn run: E " + file_name)

            env.save_recorded_episodes(file_name)
            # ===============================================

            endTime = time.time()
            total_train_time_sec = round((endTime - startTime))
            total_train_time_min = round((endTime - startTime) / 60, 2)
            time_per_episode_sec = round(total_train_time_sec / EPISODES, 3)

            # Update Results DataFrames:
            res = res.append(
                {"Model Name": MODEL_NAME, "Convolution Layers": m["conv_list"], "Dense Layers": m["dense_list"],
                 "Batch Size": m["MINIBATCH_SIZE"], "ECC": m["ECC_Settings"], "EF": m["EF_Settings"],
                 "Best Only": m["best_only"], "Average Reward": average_reward,
                 "Best Average": avg_reward_info[-1][1], "Epsilon 4 Best Average": avg_reward_info[-1][2],
                 "Best Average On": avg_reward_info[-1][0], "Max Reward": max_reward_info[-1][1],
                 "Epsilon 4 Max Reward": max_reward_info[-1][2], "Max Reward On": max_reward_info[-1][0],
                 "Total Training Time (min)": total_train_time_min, "Time Per Episode (sec)": time_per_episode_sec}
                , ignore_index=True)
            res = res.sort_values(by='Best Average')
            avg_df = pd.DataFrame(data=avg_reward_info, columns=["Episode", "Average Reward", "Epsilon"])
            max_df = pd.DataFrame(data=max_reward_info, columns=["Episode", "Max Reward", "Epsilon"])

            # Save dataFrames
            res.to_csv(f"{self.PATH}results/Results.csv")
            avg_df.to_csv(f"{self.PATH}results/{MODEL_NAME}-Results-Avg.csv")
            max_df.to_csv(f"{self.PATH}results/{MODEL_NAME}-Results-Max.csv")

        ######################################################################################

        TendTime = time.time()
        print(f"Training took {round((TendTime - TstartTime) / 60)} Minutes ")
        print(f"Training took {round((TendTime - TstartTime) / 3600)} Hours ")

        log_debug("dqn-cnn run: Z ")

        result = {"status": "ok"}
        return result

    def get_recorded_episodes(self, dic):
        print("90500-450: \n", "="*50, "\n", dic, "\n", "="*50)
        i = int(dic['i'])
        env = Environment()
        env.MOVE_WALL_EVERY = 1  # Every how many frames the wall moves.

        file_name = f'{self.PATH}pickles/{"result_for_M" + str(i) + ".pkl"}'
        result = env.get_recorded_episodes(file_name)
        # ======================

        result = {"status": "ok", "result": result}

        print(result)

        return result



