from stable_baselines3 import PPO
from rl_env_stage31 import A1Env
import mujoco.viewer
import time
import numpy as np

env = A1Env()
env.set_cmd_vel(0.30)   # slow — matches stage31 fixed cmd_vel

model = PPO.load("stage31_trot_2M")

obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    episode      = 0
    total_reward = 0.0

    while viewer.is_running():

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if env.steps % 30 == 0:
            contacts = env._get_foot_contacts()
            fr  = "X" if contacts[0] > 0.5 else "."
            fl  = "X" if contacts[1] > 0.5 else "."
            rr  = "X" if contacts[2] > 0.5 else "."
            rl  = "X" if contacts[3] > 0.5 else "."
            fwd = env.data.qvel[0]
            air = int(np.sum(1.0 - contacts))
            ph  = np.sin(env.phase_t)

            if fr=="X" and fl=="." and rr=="." and rl=="X":
                state = "TROT-A ✓"
            elif fr=="." and fl=="X" and rr=="X" and rl==".":
                state = "TROT-B ✓"
            elif fr=="X" and fl=="X" and rr=="." and rl==".":
                state = "LATERAL-FRONT ✗"
            elif fr=="." and fl=="." and rr=="X" and rl=="X":
                state = "LATERAL-REAR ✗"
            elif air == 0:
                state = "ALL-STANCE ✗"
            elif air == 4:
                state = "ALL-AIR ✗"
            else:
                state = "MIXED"

            print(
                f"FR:{fr} FL:{fl} RR:{rr} RL:{rl}  "
                f"fwd:{fwd:.3f}  air:{air}  "
                f"phase:{ph:+.2f}  {state}"
            )

        viewer.sync()
        time.sleep(0.02)

        if terminated or truncated:
            episode += 1
            print(f"Episode {episode} | total reward: {total_reward:.2f}")
            print("-" * 60)
            total_reward = 0.0
            obs, _ = env.reset()