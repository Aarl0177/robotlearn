# src/eval.py
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import imageio

from agilerl.algorithms.maddpg import MADDPG
from agilerl.utils.algo_utils import obs_channels_to_first

from src.utils.env_utils import make_simple_spread_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gif", type=str, default="results/gifs/episode.gif")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Render RGB frames
    env = make_simple_spread_env(N=3, max_cycles=args.max_steps, continuous_actions=True, render_mode="rgb_array")
    env.reset()
    agent_ids = env.agents

    maddpg = MADDPG.load(args.model, device)

    frames = []
    episodic_returns = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        score = {aid: 0.0 for aid in agent_ids}
        for t in range(args.max_steps if hasattr(args, "max-steps") else args.max_steps):
            # get action and step env
            action, _ = maddpg.get_action(obs=obs, infos=info)
            frame = env.render()  # rgb array
            frames.append(frame)
            obs, reward, termination, truncation, info = env.step({aid: action[aid].squeeze() for aid in agent_ids})
            for aid, r in reward.items():
                score[aid] += float(r)
            if any(termination.values()) or any(truncation.values()):
                break
        episodic_returns.append(sum(score.values()))

    env.close()
    os.makedirs(os.path.dirname(args.gif), exist_ok=True)
    imageio.mimsave(args.gif, frames, duration=0.03)
    print(f"Saved GIF to {args.gif}")
    print("Episode returns:", episodic_returns)

if __name__ == "__main__":
    main()
