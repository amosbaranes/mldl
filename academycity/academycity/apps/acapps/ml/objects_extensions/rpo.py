import os
#
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from ....core.utils import log_debug, clear_log_debug
#
from ....core.utils import log_debug, clear_log_debug
#
import pandas as pd
import numpy as np
#
import random
import torch
import torch.nn as nn
import gymnasium as gym
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from IPython.display import HTML, clear_output
from base64 import b64encode
#
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what is available
import matplotlib.pyplot as plt

#
class RPOAnalysis(object):
    def __init__(self, dic):
        # print("WBAnalysis\n", dic)
        try:
            self.datadir = dic['datadir']
        except Exception as ex:
            print("Error 20-01", ex, "need to provide dir name")
            self.datadir = ""
        try:
            self.model_name = dic['model_name']
        except Exception as ex:
            print("Error 20-02", ex, "need to provide model name")
            self.model_name = "General_name"
        self.checkpoint_file = os.path.join(self.datadir, "checkpoint_"+self.model_name+"_wt")
        log_debug("train_wb 110:" + self.checkpoint_file)
        # print(self.checkpoint_file)
        # ---
        self.device = None
        self.model = None
        self.env = dic["env"]
        self.n_actions = dic["n_actions"]
        self.state_dim = dic["state_dim"]
        self.seed = dic["seed"]
        self.get_model()
        print("Obj creation - model was created")
        # ---
        self.trainData = None
        self.testData = None
        # =--
        self.history = None
        log_debug("train_rpo 120:")
        print("End creation of RPOAnalysis")

    def generate_trajectory(self, n_steps=1000):
        """
        Play a session and generate a trajectory
        returns: arrays of states, actions, rewards
        """
        states, actions, rewards = [], [], []

        # initialize the environment
        s, _ = self.env.reset()

        # generate n_steps of trajectory:
        for t in range(n_steps):
            action_probs = self.predict_probs(np.array([s]))[0]
            # sample action based on action_probs
            a = np.random.choice(self.n_actions, p=action_probs)
            next_state, r, done, _, _ = self.env.step(a)

            # update arrays
            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = next_state
            if done:
                break

        return np.array(states), np.array(actions), np.array(rewards)

    def evaluate(self, n_games=3, t_max=10000):
        rewards = []
        for i in range(n_games):
            s, _ = self.env.reset(seed=self.seed + i)
            reward = 0
            for _ in range(t_max):
                action_probs = self.predict_probs(np.array([s]))[0]
                # sample action based on action_probs
                a = np.random.choice(self.n_actions, p=action_probs)
                next_state, r, terminated, _, _ = self.env.step(a)
                reward += r
                s = next_state
                if terminated:
                    break
            rewards.append(reward)
        return np.mean(rewards)

    def smoothen(self, values):
        kernel = gaussian(100, std=100)
        kernel = kernel / np.sum(kernel)
        return convolve(values, kernel, 'valid')

    def get_rewards_to_go(self, rewards, gamma=0.99):
        T = len(rewards)  # total number of individual rewards
        # empty array to return the rewards to go
        rewards_to_go = [0] * T
        rewards_to_go[T - 1] = rewards[T - 1]
        for i in range(T - 2, -1, -1):  # go from T-2 to 0
            rewards_to_go[i] = gamma * rewards_to_go[i + 1] + rewards[i]
        return rewards_to_go

    def train_one_episode(self, states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # get rewards to go
        rewards_to_go = self.get_rewards_to_go(rewards, gamma)

        # convert numpy array to torch tensors
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards_to_go = torch.tensor(rewards_to_go, device=self.device, dtype=torch.float)

        # get action probabilities from states
        logits = self.model(states)
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)

        log_probs_for_actions = log_probs[range(len(actions)), actions]

        # Compute loss to be minimized
        J = torch.mean(log_probs_for_actions * rewards_to_go)
        H = -(probs * log_probs).sum(-1).mean()

        loss = -(J + entropy_coef * H)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().cpu()  # to show progress on training

    def save(self):
        pass
        # tf.keras.models.save_model(self.model, self.checkpoint_file, overwrite=True)

    # def checkpoint_model(self):
    #     if not os.path.exists(self.checkpoint_file):
    #         # self.model.predict(np.ones((20, 28, 28), dtype=np.float32))
    #         self.save()
    #     else:
    #         self.model = tf.keras.models.load_model(self.checkpoint_file)

    def get_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.state_dim,192),
            nn.ReLU(),
            nn.Linear(192, self.n_actions),
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = self.model.to(self.device)
        # ---
        # self.checkpoint_model()
        # ---
    # --- End Model ---

    def predict_probs(self, states):
        """
        params: states: [batch, state_dim]
        returns: probs: [batch, n_actions]
        """
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(states)
        probs = nn.functional.softmax(logits, -1).detach().cpu().numpy()
        return probs

    def get_convergence_history(self, metric_name):
        # print(metric_name)
        # print(self.history.epoch, self.history.history[metric_name])
        y = self.history.history[metric_name]
        y = [round(1000*h)/1000 for h in y]
        return {"x": self.history.epoch, "y": y}

    def train(self, dic):
        log_debug("train_rpo 122: got data.")

        eval_freq = dic["eval_freq"]
        n_episodes = dic["n_episodes"]
        loss_history = []
        return_history = []
        return_history_mean = []
        evaluate_history = []

        for i in range(30000):
            states, actions, rewards = self.generate_trajectory(n_episodes)
            loss = self.train_one_episode(states, actions, rewards)
            return_history.append(np.sum(rewards))
            loss_history.append(loss)

            if i != 0 and i % eval_freq == 0:
                mean_return = np.mean(return_history[-eval_freq:])
                try:
                    print(round(100*mean_return)/100)
                    return_history_mean.append(round(100*mean_return)/100)
                except Exception as ex:
                    pass
                if mean_return > 500:
                    break

            if i != 0 and i % eval_freq == 0:
                # eval the agent
                evaluate_history.append(
                    self.evaluate()
                )
                clear_output(True)

        return_history = [round(1000*x)/100 for x in return_history]
        print("\nreturn_history_mean\n", return_history_mean)

        evaluate_history = [round(1000*x)/100 for x in evaluate_history]
        print("\nevaluate_history\n", evaluate_history)

        loss_history = [round(1000*x.item())/100 for x in loss_history]
        print("\nLoss_history\n", loss_history)


                # plt.figure(figsize=[16, 5])
                # plt.subplot(1, 2, 1)
                # plt.title("Mean return per episode")
                # plt.plot(return_history)
                # plt.grid()
                #
                # assert not np.isnan(loss_history[-1])
                # plt.subplot(1, 2, 2)
                # plt.title("Loss history (smoothened)")
                # plt.plot(smoothen(loss_history))
                # plt.grid()
                #
                # plt.show()

        return {}

def make_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    return env

# Based on Chapter 11:
# --------------------------
class RPOAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(RPOAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)

        self.app = dic["app"]


class RPODataProcessing(BaseDataProcessing, BasePotentialAlgo, RPOAlgo):
    def __init__(self, dic):
        # print("90567-010 DataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DataProcessing ", self.app)
        self.PATH = os.path.join(self.TO_OTHER, "tdqn")
        os.makedirs(self.PATH, exist_ok=True)
        # print(f'{self.PATH}')
        self.model = None
        self.state_shape = None
        self.n_actions = None
        self.state_dim = None
        self.seed = 123
        clear_log_debug()
        #

    def train(self, dic):
        print("5050-50-rpo: \n", "="*50, "\n", dic, "\n", "="*50)

        env_name = 'CartPole-v1'
        env = make_env(env_name)
        env.reset(seed=self.seed)
        # plt.imshow(env.render())
        self.state_shape, self.n_actions = env.observation_space.shape, env.action_space.n
        self.state_dim = self.state_shape[0]
        print(f"state shape:{self.state_shape}\nNumber of Actions:{self.n_actions}")

        dic["datadir"] = self.MODELS_PATH
        dic["model_name"] = "rpo_analysis"
        dic["env"] = env
        dic["state_dim"] = self.state_dim
        dic["n_actions"] = self.n_actions
        dic["seed"] = self.seed

        rpoa = RPOAnalysis(dic)
        dic["eval_freq"] = 50
        dic["n_episodes"] = 1000
        rpoa.train(dic)

        result = {"status": "ok rpo"}
        return result

