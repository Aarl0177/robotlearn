# train_maddpg.py
import os
from copy import deepcopy

import numpy as np
import torch
import supersuit as ss
from tqdm import trange

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.utils.utils import create_population
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

from env_drone_formation import make_env, FormationConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- environment config ---
    n_agents = 3
    cfg = FormationConfig(n_agents=n_agents, max_steps=600)

    # build N vectorized envs for throughput
    num_envs = 8
    env_fns = [make_env(n_agents=n_agents, seed=42 + i, cfg=cfg) for i in range(num_envs)]
    env = AsyncPettingZooVecEnv(env_fns)

    # (optional) normalize observations
    env = ss.normalize_obs_v0(env, env_min=-5, env_max=5)

    env.reset()

    # --- MADDPG networks & hyperparams ---
    NET_CONFIG = {
        "encoder_config": None,
        "head_config": {"hidden_size": [128, 128]},
    }
    INIT_HP = {
        "POPULATION_SIZE": 1,
        "ALGO": "MADDPG",
        "BATCH_SIZE": 256,
        "O_U_NOISE": True,
        "EXPL_NOISE": 0.1,
        "MEAN_NOISE": 0.0,
        "THETA": 0.15,
        "DT": 0.01,
        "LR_ACTOR": 1e-3,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.95,
        "MEMORY_SIZE": 200_000,
        "LEARN_STEP": 100,
        "TAU": 0.01,
    }

    observation_spaces = [env.single_observation_space(a) for a in env.agents]
    action_spaces = [env.single_action_space(a) for a in env.agents]
    INIT_HP["AGENT_IDS"] = deepcopy(env.agents)

    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=3e-3),
        lr_critic=RLParameter(min=1e-4, max=3e-3),
        batch_size=RLParameter(min=64, max=1024, dtype=int),
        learn_step=RLParameter(min=20, max=400, dtype=int),
    )

    agent = create_population(
        INIT_HP["ALGO"],
        observation_spaces,
        action_spaces,
        NET_CONFIG,
        INIT_HP,
        hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )[0]

    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # --- training loop ---
    max_env_steps = 200_000
    learning_starts = 2_000
    log_every = 10_000
    total_steps = 0

    print("Training MADDPG on DroneFormationEnv ...")
    pbar = trange(max_env_steps, unit="env-step")

    while total_steps < max_env_steps:
        state, info = env.reset()
        done_any = {aid: False for aid in env.agents}
        ep_return = 0.0

        while not any(done_any.values()):
            cont_actions, _ = agent.get_action(obs=state, training=True, infos=info)
            next_state, reward, termination, truncation, info = env.step(cont_actions)
            done_any = {aid: termination[aid] or truncation[aid] for aid in env.agents}

            memory.save_to_memory(state, cont_actions, reward, next_state, done_any, is_vectorised=True)

            state = next_state
            total_steps += num_envs
            ep_return += float(np.mean(list(reward.values())))

            if total_steps > learning_starts and (total_steps % INIT_HP["LEARN_STEP"] == 0):
                agent.learn(memory, learner_iterations=1)

        if total_steps % log_every == 0:
            print(f"steps={total_steps:,}  avg_ep_return={ep_return:.3f}")
        pbar.update(num_envs)

    # save checkpoint
    os.makedirs("models/MADDPG", exist_ok=True)
    path = "models/MADDPG/drone_formation.pt"
    agent.save_checkpoint(path)
    print(f"Saved checkpoint to {path}")

    env.close()


if __name__ == "__main__":
    main()