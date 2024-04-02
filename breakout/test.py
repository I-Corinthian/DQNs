import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

a2c_path = os.path.join('training','saved_model','saved_model')

model = A2C.load(a2c_path)

env = make_atari_env('Breakout-v4',n_envs=1,seed=0)
env = VecFrameStack(env,n_stack=4)

observation = env.reset()

for _ in range(100000):
    action, _= model.predict(observation)
    observation, reward, done, info= env.step(action)
    env.render(mode='human')

env.close() 