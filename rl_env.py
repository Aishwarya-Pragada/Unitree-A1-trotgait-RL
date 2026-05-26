import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class A1Env(gym.Env):

    def __init__(self):

        super().__init__()

        # =====================================================
        # LOAD MODEL
        # =====================================================

        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        self.data = mujoco.MjData(self.model)

        # =====================================================
        # ACTION SPACE
        # =====================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        # =====================================================
        # OBSERVATION SPACE
        # =====================================================

        # 47 observations
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(47,),
            dtype=np.float32
        )

        # =====================================================
        # DEFAULT STANDING POSE
        # =====================================================

        self.base_pose = np.array([
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8
        ])

        # =====================================================
        # PREVIOUS ACTION
        # =====================================================

        self.prev_action = np.zeros(12, dtype=np.float32)

    # =========================================================
    # OBSERVATION FUNCTION
    # =========================================================

    def _get_obs(self):

        # JOINT STATES
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]

        # BASE STATES
        base_height = np.array([self.data.qpos[2]])

        base_orientation = self.data.qpos[3:7]

        base_linear_velocity = self.data.qvel[0:3]

        base_angular_velocity = self.data.qvel[3:6]

        # OBSERVATION VECTOR
        obs = np.concatenate([
            base_height,
            base_orientation,
            base_linear_velocity,
            base_angular_velocity,
            joint_pos,
            joint_vel,
            self.prev_action
        ])

        return obs.astype(np.float32)

    # =========================================================
    # RESET FUNCTION
    # =========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.steps = 0

        self.prev_action = np.zeros(12, dtype=np.float32)

        obs = self._get_obs()

        return obs, {}

    # =========================================================
    # STEP FUNCTION
    # =========================================================

    def step(self, action):

        # =====================================================
        # STORE OLD ACTION FOR SMOOTHNESS PENALTY
        # =====================================================

        old_action = self.prev_action.copy()

        # =====================================================
        # CLIP ACTIONS
        # =====================================================

        action = np.clip(action, -1.0, 1.0)

        # =====================================================
        # STORE CURRENT ACTION
        # =====================================================

        self.prev_action = action.copy()

        # =====================================================
        # CONVERT ACTION TO TARGET JOINT POSITIONS
        # =====================================================

        target = self.base_pose + 0.04 * action

        # =====================================================
        # MUJOCO INTERNAL PD CONTROL
        # =====================================================

        self.data.ctrl[:] = target

        self.steps += 1

        # =====================================================
        # STEP PHYSICS
        # =====================================================

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # =====================================================
        # GET OBSERVATION
        # =====================================================

        obs = self._get_obs()

        # =====================================================
        # REWARD FUNCTION
        # =====================================================

        qvel = self.data.qvel[6:]

        # -----------------------------------------------------
        # 1. FORWARD VELOCITY REWARD
        # -----------------------------------------------------

        forward_velocity = np.clip(
            self.data.qvel[0],
            -1.0,
            1.0
        )

        forward_reward = 8.0 * forward_velocity

        # -----------------------------------------------------
        # 2. UPRIGHT STABILITY REWARD
        # -----------------------------------------------------

        # Quaternion orientation
        quat = self.data.qpos[3:7]

        # Penalize tilt
        roll_pitch_penalty = (
            abs(quat[1]) +
            abs(quat[2])
        )

        stability_reward = 1.0 - roll_pitch_penalty

        # -----------------------------------------------------
        # 3. HEIGHT STABILITY
        # -----------------------------------------------------

        height = self.data.qpos[2]

        height_penalty = abs(height - 0.27)

        # -----------------------------------------------------
        # 4. ENERGY / CONTROL PENALTY
        # -----------------------------------------------------

        control_penalty = (
            0.01 * np.sum(action ** 2)
        )

        # -----------------------------------------------------
        # 5. JOINT VELOCITY PENALTY
        # -----------------------------------------------------

        velocity_penalty = (
            0.001 * np.sum(qvel ** 2)
        )

        # -----------------------------------------------------
        # 6. SMOOTH MOTION PENALTY
        # -----------------------------------------------------

        smoothness_penalty = (
            0.05 * np.sum(
                (action - old_action) ** 2
            )
        )

        # -----------------------------------------------------
        # 7. SIDEWAYS MOTION PENALTY
        # -----------------------------------------------------

        sideways_penalty = (
            0.5 * abs(self.data.qvel[1])
        )

        # -----------------------------------------------------
        # 8. YAW INSTABILITY PENALTY
        # -----------------------------------------------------

        yaw_penalty = (
            0.1 * abs(self.data.qvel[5])
        )

        # =====================================================
        # FINAL REWARD
        # =====================================================

        reward = (
            forward_reward
            + stability_reward
            - height_penalty
            - control_penalty
            - velocity_penalty
            - smoothness_penalty
            - sideways_penalty
            - yaw_penalty
        )

        # =====================================================
        # TERMINATION CONDITIONS
        # =====================================================

        terminated = False

        # FALL DETECTION
        if self.data.qpos[2] < 0.15:
            terminated = True

        # MAX STEPS
        truncated = self.steps > 1000

        return obs, reward, terminated, truncated, {}