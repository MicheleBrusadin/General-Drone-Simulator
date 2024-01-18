import os
import datetime 

import torch
from stable_baselines3 import A2C, DQN, HER, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold

from src.drone_env_rand import DroneEnv
from src.utils import read_config
from src.monitor import Monitor
from src.logger_callback import LoggerCallback
from src.human import Human

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Read config and set up tensorboard logging
config = read_config("config_fixed_min3.yaml")
filename = "PPO_20240112-182925_fixed_min3"
# PPO_20240111-122948_base
# PPO_20240111-115042_rand
# PPO_20240112-182925_fixed_min3
# PPO_20240112-171807_rand3


env = DroneEnv(config, render_mode="human", max_episode_steps=1000)
total_score = 0
i = 0
try:
    while True and i <500:
        model = PPO.load(os.path.join('training', 'saved_models', filename), env=env)
        obs, _ = env.reset()
        done = False
        score = 0 
        
        
        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action) # Get new set of observations
            score+=reward
            
            mass = env.get_mass()
        print(f'Score: {round(score,2)}')
        print('Mass:{}'.format(round(mass,2)))
        if score>0:    
            total_score += score
            i += 1

except KeyboardInterrupt:
    print("Shutting down...")
finally:
    env.close()
    print('total_score:{}'.format(round(total_score/i,2)))
    exit()