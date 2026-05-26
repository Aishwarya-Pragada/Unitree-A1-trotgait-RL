import mujoco
import mujoco.viewer
import numpy as np

# LOAD MODEL
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# STANDING CONTROL TARGETS
stand_ctrl = np.array([
    0.0, 0.7, -1.4,   # FR
    0.0, 0.7, -1.4,   # FL
    0.0, 0.7, -1.4,   # RR
    0.0, 0.7, -1.4    # RL
])

# RESET TO HOME KEYFRAME
mujoco.mj_resetDataKeyframe(model, data, 0)

# SIMULATION LOOP
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Apply standing posture
        data.ctrl[:] = stand_ctrl

        # Step simulation
        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()