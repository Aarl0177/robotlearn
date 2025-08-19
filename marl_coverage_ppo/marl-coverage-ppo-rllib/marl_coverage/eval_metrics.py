import argparse, numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from pettingzoo.mpe import simple_spread_v3
import supersuit as ss

def make_env(n_agents=3):
    env = simple_spread_v3.env(N=n_agents, local_ratio=0.5, max_cycles=200, continuous_actions=False)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--agents", type=int, default=9)
    args = ap.parse_args()

    algo = Algorithm.from_checkpoint(args.checkpoint)
    env = make_env(args.agents)

    collisions = []
    steps = []
    for ep in range(args.episodes):
        env.reset(seed=ep)
        ep_col = 0; ep_steps = 0; done = False
        while not done:
            obs, _ = env.last()
            action = algo.compute_single_action(obs)
            _, reward, term, trunc, _ = env.step(action)
            if reward < -0.5:  # crude collision proxy
                ep_col += 1
            ep_steps += 1
            done = any(term.values()) or any(trunc.values())
        collisions.append(ep_col)
        steps.append(ep_steps)

    print(f"Episodes: {args.episodes}")
    print(f"Mean collisions/ep: {np.mean(collisions):.2f} ± {np.std(collisions):.2f}")
    print(f"Mean steps/ep: {np.mean(steps):.1f} ± {np.std(steps):.1f}")

if __name__ == "__main__":
    main()
