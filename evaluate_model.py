
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.drone_env_rand import DroneEnv
from src.utils import read_config
rand = "PPO_20240112-171807_rand3"
fixed = "PPO_20240112-182925_fixed_min3"
config_rand ="config_rand3.yaml"
config_fixed = "config_fixed_min3.yaml"
for i in range(4):
    if i<2:
        mod = rand
        if i == 0:
            conf = config_fixed
        elif i == 1:
            conf = config_rand
    else:
        mod = fixed
        if i == 2:
            conf = config_fixed
        elif i == 3:
            
            conf = config_rand
        




    config = read_config(conf)
    env = DroneEnv(config, max_episode_steps=1000)
    model = PPO.load(os.path.join('training', 'saved_models', mod), env=env)

    # PPO_20240111-122948_base
    # PPO_20240111-115042_rand
    

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5000)

    # Print the results
    print(f"Model: {mod} , Configuration {conf} Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    env.close()

