import random
from threading import Lock

import torch
import numpy as np
from collections import deque
from src.model import Linear_QNet, QTrainer
from src.drone import Drone

class Agent:
    def __init__(self, config: dict):
        self.max_x = config['display']['width']
        self.max_y = config['display']['height']
        
        self.n_games = 0
        self.epsilon = config['agent']['epsilon']
        self.epsilon_decay = config['agent']['epsilon_decay']
        self.gamma = config['agent']['gamma']
        self.memory = deque(maxlen=config["agent"]["max_memory"])
        self.batch_size = config["agent"]["batch_size"]

        STATE_SPACE_SIZE = 6
        self.layers = [STATE_SPACE_SIZE] + config["agent"]["layers"] + [len(config["drone"]["motors"])]
        print("Layers:", self.layers)

        # Check for GPU availability and use GPU if available, else CPU
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model initialization
        self.model = Linear_QNet(layers=self.layers, dropout_p=config["agent"]["dropout_p"])
        self.model.to(self.device)  # Move the model to the specified device (GPU or CPU)
        
        # Trainer initialization
        self.trainer = QTrainer(self.model, lr=config['agent']['learning_rate'], gamma=self.gamma, device=self.device)


        # Support safe mutlithreading
        self.memory_lock = Lock()
        self.model_lock = Lock()

    def train_long_memory(self):
        with self.memory_lock:
            mini_sample = random.sample(self.memory, self.batch_size) if len(self.memory) > self.batch_size else list(self.memory)

        self._train_mini_batch(mini_sample)

    def _train_mini_batch(self, mini_sample):
        with self.model_lock:
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, previous_state, action, reward, state, done):
        with self.model_lock:
            self.trainer.train_step(previous_state, action, reward, state, done)
            
    def remember(self, state, action, reward, next_state, done):
        with self.memory_lock:
            self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state: list):
        # random moves: tradeoff exploration / exploitation
        self.epsilon -= self.epsilon_decay

        action = [0] * self.layers[-1]
        # Action is a list of n values (last layer size)
        if random.random() < self.epsilon:
            # List of random values between 0 and 1 of length last layer size
            action = np.random.rand(self.layers[-1])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state_tensor)
            action = prediction.detach().cpu().numpy()  # Move back to CPU if needed
    
        return action

    def get_reward(self, state: list, target: dict, done: bool):
        # Dying is bad
        if done:
            return -10

        # Calculate Euclidean distance from the target
        distance = np.sqrt((target["x"] - state[0]) ** 2 + (target["y"] - state[1]) ** 2)
        #distance_reward = max(1 - distance/target["distance"], -2)

        if distance < target["distance"]:
            # Add a reward for being close to the target
            distance_reward = 1
        else:
            # Add a penalty for being far from the target
            distance_reward =  -0.2 * distance/target["distance"]

        # Add a constant reward for survival/progress
        constant_reward = 0.2

        return constant_reward + distance_reward



    def load(self):
        pass
