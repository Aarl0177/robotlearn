# marl_coverage/rollout.py
import argparse, os, imageio, numpy as np
import supersuit as ss
from urllib.parse import urlparse, unquote

# Try PettingZoo MPE; fall back to mpe2 if needed
try:
    from pettingzoo.mpe import simple_spread_v3 as _simple_spread_v3
except Exception:
    _simple_spread_v3 = None

from ray.rllib.algorithms.algorithm import Algorithm

def _normalize_ckpt_path(p: str) -> str:
    """Accept RLlib 'file:///' URIs or normal paths and return a local path."""
    if p.startswith("file://"):
        u = urlparse(p)
        path = unquote(u.path)
        # On Windows, strip leading slash from "/C:/..."
        if os.name == "nt" and path.startswith("/"):
            path = path[1:]
        return path.replace("/", os.sep)
    return p

def make_env(n_agents=3):
    """Create a renderable PettingZoo env with padding (as used in training)."""
    if _simple_spread_v3 is None:
        from mpe2 import simple_spread_v3 as _ssv3
        env = _ssv3.env(N=n_agents, local_ratio=0.5, max_cycles=200,
                        continuous_actions=False, render_mode="rgb_array")
    else:
        env = _simple_spread_v3.env(N=n_agents, local_ratio=0.5, max_cycles=200,
                                    continuous_actions=False, render_mode="rgb_array")
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path/URI printed by training (algo.save())")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--gif", type=str, default="out/rollout.gif")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--skip", type=int, default=1, help="Save every k-th frame to keep GIF small")
    args = ap.parse_args()

    ckpt = _normalize_ckpt_path(args.checkpoint)
    algo = Algorithm.from_checkpoint(ckpt)

    # Get the policy (we trained a single shared policy)
    try:
        policy = algo.get_policy("shared_policy")
    except Exception:
        policy = algo.get_policy()  # fallback name in some builds

    os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)

    env = make_env(args.agents)

    all_frames = []
    for ep in range(args.episodes):
        env.reset(seed=ep)
        # Standard AEC loop: step through each agent in sequence
        frames = []
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                # Compute action from the trained policy
                # (policy expects a single observation tensor)
                action, _, _ = policy.compute_single_action(obs)
            env.step(action)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Subsample frames if requested (to make GIF lighter)
        if args.skip > 1:
            frames = frames[::args.skip]
        all_frames += frames

    # Write the GIF
    if len(all_frames) == 0:
        raise RuntimeError("No frames captured. Make sure render_mode='rgb_array' and imageio is installed.")
    imageio.mimsave(args.gif, all_frames, fps=args.fps)
    print(f"Saved GIF to {args.gif}  (frames={len(all_frames)}, fps={args.fps}, skip={args.skip})")

if __name__ == "__main__":
    main()
