from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback
)

from rl_env_stage2 import A1Env


class CurriculumCallback(BaseCallback):

    def __init__(self, advance_every=100_000, verbose=0):
        super().__init__(verbose)
        self.advance_every = advance_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.advance_every == 0 and self.num_timesteps > 0:
            self.training_env.env_method("advance_curriculum")
            print(
                f"[Curriculum] step={self.num_timesteps} "
                f"-> cmd_vel advanced"
            )
        return True


env = A1Env()


model = PPO.load("stage1_trot_1M", env=env)

print("Loaded stage1_trot_1M successfully.")
print("Continuing training with stage2 environment fixes...")

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="stage2_trot"
)

curriculum_callback = CurriculumCallback(
    advance_every=100_000
)

model.learn(
    total_timesteps=1_000_000,
    callback=[checkpoint_callback, curriculum_callback],
    reset_num_timesteps=False   # keeps timestep counter continuing from 1M
)

model.save("stage2_trot_2M")

print("Stage 2 training completed.")
print("Saved to: stage2_trot_2M")
