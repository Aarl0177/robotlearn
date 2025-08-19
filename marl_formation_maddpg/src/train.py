# src/train.py
from __future__ import annotations
import argparse, os, json, time
from copy import deepcopy

import numpy as np
import torch
from tqdm import trange

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.utils.utils import create_population

from src.utils.env_utils import make_simple_spread_env
from src.utils.plot import plot_learning_curves

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--n-envs", type=int, default=1)  # keep 1 for simplicity
    parser.add_argument("--save-dir", type=str, default="runs/exp1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ====== 1) ENV ======
    env = make_simple_spread_env(N=3, max_cycles=args.max_steps, local_ratio=0.5,
                                 continuous_actions=True, render_mode=None)
    obs, info = env.reset(seed=args.seed)
    agent_ids = deepcopy(env.agents)

    # ====== 2) MADDPG config ======
    # Vector observations, so we only need MLP "head_config"
    NET_CONFIG = {"head_config": {"hidden_size": [128, 128]}}

    INIT_HP = {
        "POPULATION_SIZE": 1,
        "ALGO": "MADDPG",
        "CHANNELS_LAST": False,  # no images here
        "BATCH_SIZE": 256,
        "O_U_NOISE": True,
        "EXPL_NOISE": 0.1,
        "MEAN_NOISE": 0.0,
        "THETA": 0.15,
        "DT": 0.01,
        "LR_ACTOR": 1e-3,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.95,
        "MEMORY_SIZE": 200000,
        "LEARN_STEP": 100,
        "TAU": 0.01,
        "AGENT_IDS": agent_ids,
    }

    # Observation / action spaces (same across agents in MPE)
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # Simple HP search space (you can ignore; keeps API happy)
    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=64, max=512, dtype=int),
        learn_step=RLParameter(min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75),
    )

    agent: MADDPG = create_population(
        INIT_HP["ALGO"],
        observation_spaces,
        action_spaces,
        NET_CONFIG,
        INIT_HP,
        hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=args.n_envs,
        device=device,
    )[0]
    agent.set_training_mode(True)

    # ====== 3) Replay buffer ======
    field_names = ["obs", "action", "reward", "next_obs", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=agent_ids,
        device=device,
    )

    # ====== 4) Train loop ======
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    models_dir = os.path.join(args.save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    global_steps = 0
    best_avg_return = -1e9
    pbar = trange(args.episodes, desc="Training (episodes)")

    for ep in pbar:
        obs, info = env.reset()
        scores = {aid: 0.0 for aid in agent_ids}

        for t in range(args.max_steps):
            # Agent chooses continuous actions (dict of np arrays per agent)
            action, raw_action = agent.get_action(obs=obs, infos=info)

            # Step env
            next_obs, reward, termination, truncation, info = env.step(
                {aid: action[aid].squeeze() for aid in agent_ids}
            )

            # Save to replay buffer
            memory.save_to_memory(obs, raw_action, reward, next_obs, termination, is_vectorised=False)

            # Learning step
            if len(memory) >= agent.batch_size and memory.counter > 500 and (global_steps % agent.learn_step == 0):
                experiences = memory.sample(agent.batch_size)
                agent.learn(experiences)

            # Accumulate rewards
            for aid, r in reward.items():
                scores[aid] += float(r)

            obs = next_obs
            global_steps += 1
            if any(termination.values()) or any(truncation.values()):
                break

        # Episode summary
        ep_return = float(sum(scores.values()))
        agent.scores.append(ep_return)
        agent.steps.append(global_steps)

        # periodic eval-style average over rolling window
        window = 10
        last = agent.scores[-window:] if len(agent.scores) >= window else agent.scores
        avg_ret = float(np.mean(last))
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"episode": ep+1, "global_step": global_steps, "return": ep_return, "avg_return": avg_ret}) + "\n")
        #pbar.set_postfix(return=ep_return, avg_return=avg_ret)
        pbar.set_postfix({"ret": f"{ep_return:.2f}", "avg": f"{avg_ret:.2f}"})
        # save latest
        latest_path = os.path.join(models_dir, "MADDPG_latest.pt")
        agent.save_checkpoint(latest_path)
        if avg_ret > best_avg_return:
            best_avg_return = avg_ret
            agent.save_checkpoint(os.path.join(models_dir, "MADDPG_best.pt"))

    env.close()

    # Plot curves
    plot_learning_curves(metrics_path, os.path.join(args.save_dir, "curves.png"))
    print("Done. Metrics at:", metrics_path)

if __name__ == "__main__":
    main()
