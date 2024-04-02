import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


#step1: make the env 
env = gym.make("CartPole-v1", render_mode="human")

#traing the agent 

log_path = os.path.join('training','logs')

env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy',env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=80000)

PPO_path = os.path.join('training','saved_models','PPO_model_cartpole')

model.save(PPO_path)