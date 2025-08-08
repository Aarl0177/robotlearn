# record_fetchreach.py
import os
os.environ["MUJOCO_GL"] = "glfw"   # override the system-wide 'wgl' for this run

import gymnasium as gym, gymnasium_robotics
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC

def main():
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchReachDense-v4", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", name_prefix="fetchreach_dense", episode_trigger=lambda ep: True)

    model = SAC.load(r"models\m1_dense_20250808_131518\sac_fetchreach_dense_final.zip", env=env)
    obs, _ = env.reset(seed=42)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
    env.close()
    print("Saved to ./videos")

if __name__ == "__main__":
    main()
