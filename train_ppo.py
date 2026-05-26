from stable_baselines3 import PPO
from rl_env import A1Env

env = A1Env()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=10000)

model.save("a1_ppo_model")

print("Training completed")