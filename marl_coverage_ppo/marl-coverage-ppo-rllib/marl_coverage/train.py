# marl_coverage/train.py — RLlib new API + GIF (Windows friendly)
import argparse, os, imageio
import numpy as np
import torch
import ray
import supersuit as ss
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec

# Try PettingZoo MPE; fallback to mpe2 if needed
try:
    from pettingzoo.mpe import simple_spread_v3 as _simple_spread_v3
except Exception:
    _simple_spread_v3 = None

POLICY_ID = "shared_policy"

def make_pz_env(n_agents=3, render=False):
    """Create simple_spread; render=True => RGB frames for GIFs."""
    render_mode = "rgb_array" if render else None
    if _simple_spread_v3 is None:
        from mpe2 import simple_spread_v3 as _ssv3
        env = _ssv3.env(N=n_agents, local_ratio=0.5, max_cycles=200,
                        continuous_actions=False, render_mode=render_mode)
    else:
        env = _simple_spread_v3.env(N=n_agents, local_ratio=0.5, max_cycles=200,
                                    continuous_actions=False, render_mode=render_mode)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    return env

def _pick_metric(result: dict):
    """Try common keys for mean return in RLlib's new stack."""
    for path in [
        ("env_runners", "episode_return_mean"),
        ("episode_reward_mean",),
        ("evaluation", "episode_reward_mean"),
        ("evaluation", "episode_return_mean"),
    ]:
        cur = result
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False; break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return None

def _compute_action_with_module(algo, obs):
    """
    New-API action: use RLModule to get logits and pick an action.
    simple_spread has discrete actions.
    """
    module = algo.get_module(POLICY_ID)  # RLModule
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.ndim == 1:
        obs_arr = obs_arr[None, :]
    obs_t = torch.from_numpy(obs_arr)

    with torch.no_grad():
        out = module.forward_inference({"obs": obs_t})

    # 1) If module gives actions directly, use them.
    if isinstance(out, dict) and "actions" in out:
        act_t = out["actions"]
    else:
        # 2) Otherwise, get logits explicitly (no `or` on tensors!)
        logits = None
        if isinstance(out, dict):
            if "action_dist_inputs" in out:
                logits = out["action_dist_inputs"]
            elif "action_logits" in out:
                logits = out["action_logits"]
            elif "logits" in out:
                logits = out["logits"]
        if logits is None:
            raise KeyError(f"RLModule output does not contain actions/logits. Keys: {list(out.keys()) if isinstance(out, dict) else type(out)}")
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        if logits.ndim == 1:
            logits = logits[None, :]
        # Deterministic for GIFs: argmax. (Swap to sampling if you prefer stochastic)
        act_t = torch.argmax(logits, dim=-1)

    # Convert to plain int
    if hasattr(act_t, "detach"):
        act_t = act_t.detach().cpu().numpy()
    if isinstance(act_t, np.ndarray):
        act_t = act_t[0] if act_t.ndim > 0 else act_t.item()
    try:
        return int(act_t)
    except Exception:
        return act_t

def render_gif(algo, n_agents, episodes, gif_path, fps=15):
    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    env = make_pz_env(n_agents=n_agents, render=True)
    frames = []
    for ep in range(episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)  # advance AEC when current agent is done
            else:
                action = _compute_action_with_module(algo, obs)
                env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = any(env.terminations.values()) or any(env.truncations.values())
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"[GIF] Saved to: {gif_path}  (frames={len(frames)}, fps={fps})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--stop-iters", type=int, default=50)
    ap.add_argument("--gif", type=str, default="out/rollout.gif")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True)

    # Register training env (no render for speed)
    register_env(
        "mpe_simple_spread",
        lambda cfg: PettingZooEnv(make_pz_env(n_agents=int(cfg.get("N", 3)), render=False))
    )
    algo = (
        PPOConfig()
        .environment(env="mpe_simple_spread", env_config={"N": args.agents})
        .multi_agent(
            policies={POLICY_ID: PolicySpec()},
            policy_mapping_fn=lambda agent_id, *a, **k: POLICY_ID,
        )
        .env_runners(num_env_runners=0)   # ↑ later if you want parallelism
        .framework("torch")
        .training(model={"fcnet_hiddens":[256,256]}, lr=5e-4, gamma=0.99)
        .build()
    )

    for i in range(args.stop_iters):
        result = algo.train()
        if (i + 1) % 10 == 0:
            rew = _pick_metric(result)
            rstr = f"{rew:.2f}" if rew is not None else "n/a"
            steps = result.get("env_runners", {}).get("num_env_steps_sampled",
                     result.get("num_env_steps_sampled"))
            print(f"[Iter {i+1}] return_mean={rstr}  steps={steps}")

    # Make GIF without saving a checkpoint
    render_gif(algo, n_agents=args.agents, episodes=args.episodes, gif_path=args.gif)
    ray.shutdown()

if __name__ == "__main__":
    main()
