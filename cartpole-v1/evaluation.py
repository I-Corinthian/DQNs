import os
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")
PPO_path = os.path.join('training','saved_models','PPO_model_cartpole')

model = PPO.load(PPO_path,env=env)

#step2 env have to be reset 

for ep in range(1,10):
    observation, info = env.reset(seed=42)
    done = False
    #step3 action take pace 
    while not done:
        action, _= model.predict(observation)
        observation, reward, terminated, truncated, info, = env.step(action=action)

        if terminated or truncated:
            done = True

#step env close 
env.close()