from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback
)

from rl_env_stage31 import A1Env


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

# =====================================================
# LOAD stage3_trot_1M and continue training
#
# Robot already knows:
#   - how to stay upright
#   - basic diagonal trot pattern
#   - how to move forward
#
# Stage 31 fixes being applied on top:
#   - phase_dt: 0.015 -> 0.10  (fixes bouncing)
#   - action scale: 0.15 -> 0.30  (fixes timing mismatch)
#   - coast_penalty: 3.0 -> 1.0  (fixes jerky recovery)
#   - cmd_vel_max fixed at 0.10  (slow motion only)
#
# Robot will start from existing knowledge and refine
# — expect ep_rew_mean to start HIGHER than stage30
# =====================================================

model = PPO.load("stage3_trot_1M", env=env)

print("Loaded stage3_trot_1M successfully.")
print("Continuing training with stage31 environment fixes...")

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="stage31_trot"
)

curriculum_callback = CurriculumCallback(
    advance_every=100_000
)

model.learn(
    total_timesteps=1_000_000,
    callback=[checkpoint_callback, curriculum_callback],
    reset_num_timesteps=False   # keeps timestep counter continuing from 1M
)

model.save("stage31_trot_2M")

print("Stage 31 training completed.")
print("Saved to: stage31_trot_2M")