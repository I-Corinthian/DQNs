import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

env = gym.make("Breakout-v4", render_mode="human")
env = make_atari_env('Breakout-v4',n_envs=4,seed=0)
env = VecFrameStack(env,n_stack=4)

log_path = os.path.join('training','logs')

model = A2C('CnnPolicy',env,verbose=1,tensorboard_log=log_path)

model.learn(total_timesteps=300000)

a2c_path = os.path.join('training','saved_model')

model.save(a2c_path)