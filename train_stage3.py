from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback
)

from rl_env_stage32 import A1Env


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
# LOAD stage31_trot_2M and continue training
#
# Robot already knows from stage31:
#   - stable upright posture
#   - diagonal trot pattern
#   - forward motion
#   - basic yaw control
#
# Stage32 adds ONE new fix on top:
#   - force_symmetry_penalty 2.0
#     tracks avg forward vel during TROT-A vs TROT-B
#     penalises if one diagonal generates more
#     propulsion than the other
#
# Root cause being fixed:
#   one diagonal does all work
#   → body tips
#   → abduction joints overcompensate
#   → yaw rotation appears
# =====================================================

model = PPO.load("stage31_trot_2M", env=env)

print("Loaded stage31_trot_2M successfully.")
print("Continuing training with stage32 — force_symmetry_penalty added.")

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="stage32_trot"
)

curriculum_callback = CurriculumCallback(
    advance_every=100_000
)

model.learn(
    total_timesteps=1_000_000,
    callback=[checkpoint_callback, curriculum_callback],
    reset_num_timesteps=False   # continue counter from 2M
)

model.save("stage32_trot_3M")

print("Stage 32 training completed.")
print("Saved to: stage32_trot_3M")
