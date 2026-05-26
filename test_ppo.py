from stable_baselines3 import PPO
from rl_env import A1Env
import mujoco.viewer
import mujoco
import time

# LOAD ENVIRONMENT
env = A1Env()

# LOAD TRAINED MODEL
model = PPO.load("a1_ppo_model")

# RESET ENVIRONMENT
obs, _ = env.reset()

# OPEN MUJOCO VIEWER
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    while viewer.is_running():

        # PPO PREDICTS ACTION
        action, _ = model.predict(
            obs,
            deterministic=True
        )

        # STEP ENVIRONMENT
        obs, reward, terminated, truncated, _ = env.step(action)

        # UPDATE VIEWER
        viewer.sync()

        time.sleep(0.01)

        # RESET IF EPISODE ENDS
        if terminated or truncated:
            obs, _ = env.reset()