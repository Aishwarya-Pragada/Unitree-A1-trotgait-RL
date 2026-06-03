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
        self.data  = mujoco.MjData(self.model)

        # =====================================================
        # HOME POSE
        # =====================================================

        self.base_pose = np.array([
            0.0,  0.9, -1.8,
            0.0,  0.9, -1.8,
            0.0,  0.9, -1.8,
            0.0,  0.9, -1.8
        ], dtype=np.float32)

        # =====================================================
        # FOOT GEOM IDs
        # Order: FR=0, FL=1, RR=2, RL=3
        # =====================================================

        self.foot_geom_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        ], dtype=np.int32)

        if np.any(self.foot_geom_ids < 0):
            self.foot_geom_ids = self._find_foot_geoms()

        # =====================================================
        # FOOT BODY IDs
        # =====================================================

        self.foot_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        ], dtype=np.int32)

        # =====================================================
        # TROT PHASE CLOCK — unchanged from stage31
        # =====================================================

        self.phase_t  = 0.0
        self.phase_dt = 0.015

        # =====================================================
        # REFERENCE MOTION — unchanged from stage31
        # =====================================================

        self.ref_scale = np.zeros(12, dtype=np.float32)
        self.ref_scale[1]  =  0.35   # FR thigh
        self.ref_scale[2]  =  0.06   # FR knee
        self.ref_scale[4]  = -0.35   # FL thigh
        self.ref_scale[5]  = -0.06   # FL knee
        self.ref_scale[7]  = -0.15   # RR thigh
        self.ref_scale[8]  = -0.10   # RR knee
        self.ref_scale[10] =  0.15   # RL thigh
        self.ref_scale[11] =  0.10   # RL knee

        # =====================================================
        # TROT STATE TRACKING
        # =====================================================

        self.prev_trot_state  = -1
        self.state_A_steps    = 0
        self.state_B_steps    = 0
        self.same_state_steps = 0

        # =====================================================
        # FORCE SYMMETRY TRACKING — NEW in stage32
        #
        # Track forward velocity accumulated during
        # TROT-A and TROT-B separately.
        #
        # Root cause chain identified from stage31:
        #   one diagonal does all the work
        #   → body tips to that side
        #   → abduction joints overcompensate
        #   → yaw rotation appears
        #
        # Fixing force symmetry first will automatically
        # reduce abduction overuse and yaw as side effects.
        # =====================================================

        self.vel_A_sum = 0.0
        self.vel_B_sum = 0.0

        # =====================================================
        # CURRICULUM — unchanged from stage31
        # cmd_vel fixed at 0.30, no further increase
        # =====================================================

        self.cmd_vel      = 0.30
        self.cmd_vel_min  = 0.10
        self.cmd_vel_max  = 0.30
        self.cmd_vel_step = 0.02

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
        # OBSERVATION SPACE — 51
        # =====================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(51,),
            dtype=np.float32
        )

        self.steps         = 0
        self.max_steps     = 1000
        self.prev_action   = np.zeros(12, dtype=np.float32)
        self.prev_contacts = np.zeros(4,  dtype=np.float32)
        self.recent_z_vels = np.zeros(10, dtype=np.float32)
        self.z_vel_idx     = 0

    # =========================================================
    # FOOT GEOM FINDER (fallback)
    # =========================================================

    def _find_foot_geoms(self):

        foot_body_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        ids = []

        for bname in foot_body_names:
            bid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                bname
            )
            geom_ids_for_body = [
                g for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == bid
            ]
            ids.append(geom_ids_for_body[-1] if geom_ids_for_body else -1)

        return np.array(ids, dtype=np.int32)

    # =========================================================
    # FOOT CONTACT DETECTION
    # =========================================================

    def _get_foot_contacts(self):

        contacts = np.zeros(4, dtype=np.float32)

        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            for k, fid in enumerate(self.foot_geom_ids):
                if fid < 0:
                    continue
                if g1 == fid or g2 == fid:
                    contacts[k] = 1.0

        return contacts

    # =========================================================
    # FOOT VELOCITIES IN WORLD FRAME
    # =========================================================

    def _get_foot_velocities(self):

        vels = np.zeros((4, 3), dtype=np.float32)

        for k, bid in enumerate(self.foot_body_ids):
            if bid < 0:
                continue
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(
                self.model,
                self.data,
                jacp,
                jacr,
                bid
            )
            lin_vel = jacp @ self.data.qvel
            vels[k] = lin_vel

        return vels

    # =========================================================
    # REFERENCE MOTION
    # =========================================================

    def _get_reference_joints(self):
        phase = np.sin(self.phase_t)
        return self.base_pose + self.ref_scale * phase

    # =========================================================
    # STRICT TROT REWARD
    # =========================================================

    def _get_strict_trot_reward(self, contacts, forward_vel):

        fr = contacts[0]
        fl = contacts[1]
        rr = contacts[2]
        rl = contacts[3]

        score_A = (fr + (1.0-fl) + (1.0-rr) + rl) / 4.0
        score_B = ((1.0-fr) + fl + rr + (1.0-rl)) / 4.0

        current_state = 0 if score_A >= score_B else 1
        best_score    = max(score_A, score_B)

        if current_state == 0:
            self.state_A_steps += 1
            self.vel_A_sum     += max(0.0, forward_vel)  # NEW stage32
        else:
            self.state_B_steps += 1
            self.vel_B_sum     += max(0.0, forward_vel)  # NEW stage32

        if current_state == self.prev_trot_state:
            self.same_state_steps += 1
        else:
            self.same_state_steps = 0

        trot_reward = 2.0 * np.exp(8.0 * (best_score - 1.0))

        stuck_penalty = 0.0
        if self.same_state_steps > 15:
            trot_reward   = 0.0
            stuck_penalty = 1.50 * min(self.same_state_steps / 15.0, 3.0)

        transition_reward = 0.0
        if (self.prev_trot_state != -1 and
                current_state != self.prev_trot_state):
            transition_reward = 1.50

        self.prev_trot_state = current_state

        # symmetry penalty — unchanged from stage31
        symmetry_penalty = 0.0
        total = self.state_A_steps + self.state_B_steps
        if total > 10:
            ratio = self.state_A_steps / (total + 1e-6)
            symmetry_penalty = 4.0 * abs(ratio - 0.5)

        # =====================================================
        # NEW stage32: force symmetry penalty
        #
        # Tracks avg forward velocity during TROT-A vs TROT-B.
        # If one diagonal generates more forward velocity
        # than the other, penalise the difference.
        #
        # This directly fixes: "Coordination symmetry exists,
        # Force symmetry does not exist."
        #
        # Needs at least 5 steps in each state before
        # activating — avoids division noise at episode start.
        # =====================================================
        force_symmetry_penalty = 0.0
        if self.state_A_steps > 5 and self.state_B_steps > 5:
            avg_A = self.vel_A_sum / (self.state_A_steps + 1e-6)
            avg_B = self.vel_B_sum / (self.state_B_steps + 1e-6)
            force_symmetry_penalty = 2.0 * abs(avg_A - avg_B)

        front_wrong = fr*fl + (1-fr)*(1-fl)
        rear_wrong  = rr*rl + (1-rr)*(1-rl)
        all_stance  = fr*fl*rr*rl
        all_air     = (1-fr)*(1-fl)*(1-rr)*(1-rl)

        wrong_penalty = (
            1.50 * front_wrong
            + 1.50 * rear_wrong
            + 2.00 * all_stance
            + 2.00 * all_air
        )

        return (
            trot_reward,
            transition_reward,
            symmetry_penalty,
            force_symmetry_penalty,
            wrong_penalty,
            stuck_penalty
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_obs(self):

        base_height      = np.array([self.data.qpos[2]])
        orientation      = self.data.qpos[3:7]
        angular_velocity = self.data.qvel[3:6]
        joint_positions  = self.data.qpos[7:]
        joint_velocities = self.data.qvel[6:]
        foot_contacts    = self._get_foot_contacts()
        prev_action      = self.prev_action
        cmd_vel          = np.array([self.cmd_vel], dtype=np.float32)

        phase_sin = np.array([np.sin(self.phase_t)], dtype=np.float32)
        phase_cos = np.array([np.cos(self.phase_t)], dtype=np.float32)

        obs = np.concatenate([
            base_height,       #  1
            orientation,       #  4
            angular_velocity,  #  3
            joint_positions,   # 12
            joint_velocities,  # 12
            foot_contacts,     #  4
            prev_action,       # 12
            cmd_vel,           #  1
            phase_sin,         #  1
            phase_cos          #  1
        ])                     # = 51

        return obs.astype(np.float32)

    # =========================================================
    # RESET
    # =========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.steps            = 0
        self.prev_action      = np.zeros(12, dtype=np.float32)
        self.prev_contacts    = np.zeros(4,  dtype=np.float32)
        self.recent_z_vels    = np.zeros(10, dtype=np.float32)
        self.z_vel_idx        = 0
        self.prev_trot_state  = -1
        self.state_A_steps    = 0
        self.state_B_steps    = 0
        self.same_state_steps = 0
        self.vel_A_sum        = 0.0   # NEW stage32
        self.vel_B_sum        = 0.0   # NEW stage32

        self.phase_t = np.random.uniform(0, 2 * np.pi)

        self.data.qpos[3]  += np.random.uniform(-0.02, 0.02)
        self.data.qpos[4]  += np.random.uniform(-0.02, 0.02)
        self.data.qpos[7:] += np.random.uniform(-0.02, 0.02, size=12)

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    # =========================================================
    # CURRICULUM CONTROL
    # =========================================================

    def advance_curriculum(self):
        self.cmd_vel = min(
            self.cmd_vel + self.cmd_vel_step,
            self.cmd_vel_max
        )

    def set_cmd_vel(self, v):
        self.cmd_vel = float(np.clip(v, self.cmd_vel_min, self.cmd_vel_max))

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):

        action = np.clip(action, -1.0, 1.0)

        self.data.ctrl[:] = self.base_pose + 0.15 * action

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self.phase_t += self.phase_dt
        self.steps   += 1

        obs = self._get_obs()

        # ===================================================
        # STATE EXTRACTION
        # ===================================================

        height     = self.data.qpos[2]
        quat       = self.data.qpos[3:7]

        roll_component  = abs(quat[1])
        pitch_component = abs(quat[2])

        roll_rate    = abs(self.data.qvel[3])
        pitch_rate   = abs(self.data.qvel[4])
        yaw_rate     = abs(self.data.qvel[5])

        forward_vel  = self.data.qvel[0]
        lateral_vel  = abs(self.data.qvel[1])
        vertical_vel = self.data.qvel[2]

        action_diff   = np.sum(np.square(action - self.prev_action))

        foot_contacts  = self._get_foot_contacts()
        foot_vels      = self._get_foot_velocities()
        feet_on_ground = np.sum(foot_contacts)
        feet_in_air    = 4.0 - feet_on_ground

        self.recent_z_vels[self.z_vel_idx] = abs(vertical_vel)
        self.z_vel_idx = (self.z_vel_idx + 1) % 10
        avg_bounce     = np.mean(self.recent_z_vels)

        self.prev_action   = action.copy()
        self.prev_contacts = foot_contacts.copy()

        fwd_vel_clipped = max(0.0, forward_vel)

        # ===================================================
        # REFERENCE MOTION REWARD
        # ===================================================

        ref_joints  = self._get_reference_joints()
        actual_ctrl = self.data.ctrl[:]
        joint_error = np.sum(np.square(actual_ctrl - ref_joints))
        ref_reward  = 1.50 * np.exp(-2.0 * joint_error)

        # ===================================================
        # REWARDS — all unchanged from stage31
        # except force_symmetry_penalty added
        # ===================================================

        alive_reward   = 1.0

        upright_reward = 1.0 - roll_component - pitch_component

        height_reward  = 1.0 - abs(height - 0.27)

        oscillation_penalty = 0.10 * (roll_rate + pitch_rate)

        smoothness_penalty  = 0.05 * action_diff

        lateral_penalty     = 0.40 * lateral_vel

        roll_penalty        = 0.30 * roll_component

        bounce_penalty      = 0.50 * avg_bounce

        height_stability    = 0.30 * np.exp(-10.0 * abs(height - 0.27))

        yaw_penalty = 0.50 * yaw_rate

        early_stable_bonus = 0.0
        if self.steps <= 100:
            early_stable_bonus = (
                0.50
                * (1.0 - roll_component)
                * (1.0 - pitch_component)
                * (1.0 - abs(height - 0.27))
            )

        if feet_in_air >= 1.0:
            forward_reward = 2.0 * min(fwd_vel_clipped, self.cmd_vel) / (self.cmd_vel + 1e-6)
            coast_penalty  = 0.0
        else:
            forward_reward = 0.0
            coast_penalty  = 1.0 * fwd_vel_clipped

        slide_penalty = 0.0
        grip_reward   = 0.0
        for k in range(4):
            if foot_contacts[k] > 0.5:
                fhv = np.linalg.norm(foot_vels[k, :2])
                slide_penalty += 0.80 * fhv
                grip_reward   += 0.20 * np.exp(-5.0 * fhv)

        stance_push_reward = 0.0
        if feet_on_ground == 2.0 and forward_vel > 0.05:
            stance_push_reward = 0.50 * min(fwd_vel_clipped, self.cmd_vel) / (self.cmd_vel + 1e-6)

        trot_reward            = 0.0
        transition_reward      = 0.0
        symmetry_penalty       = 0.0
        force_symmetry_penalty = 0.0   # NEW stage32
        wrong_penalty          = 0.0
        stuck_penalty          = 0.0

        if forward_vel > 0.05:
            (
                trot_reward,
                transition_reward,
                symmetry_penalty,
                force_symmetry_penalty,
                wrong_penalty,
                stuck_penalty
            ) = self._get_strict_trot_reward(foot_contacts, forward_vel)

        reward = (
            alive_reward
            + upright_reward
            + height_reward
            + forward_reward
            + grip_reward
            + stance_push_reward
            + trot_reward
            + transition_reward
            + height_stability
            + early_stable_bonus
            + ref_reward
            - oscillation_penalty
            - smoothness_penalty
            - slide_penalty
            - lateral_penalty
            - roll_penalty
            - coast_penalty
            - bounce_penalty
            - symmetry_penalty
            - force_symmetry_penalty   # NEW stage32
            - wrong_penalty
            - stuck_penalty
            - yaw_penalty
        )

        # ===================================================
        # TERMINATION
        # ===================================================

        terminated = False

        if height < 0.18:
            terminated = True

        if roll_component > 0.60:
            terminated = True

        if pitch_component > 0.65:
            terminated = True

        truncated = (self.steps >= self.max_steps)

        return obs, reward, terminated, truncated, {}
