# evaluate.py
import numpy as np
import torch
from agilerl.utils.utils import load_checkpoint
from env_drone_formation import make_env, FormationConfig
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv


def evaluate(ckpt_path: str = "models/MADDPG/drone_formation.pt", episodes: int = 5):
    n_agents = 3
    cfg = FormationConfig(n_agents=n_agents, max_steps=600)

    env = AsyncPettingZooVecEnv([make_env(n_agents=n_agents, seed=999, cfg=cfg)])
    env.reset()

    # reload MADDPG agent
    maddpg = load_checkpoint(ckpt_path)

    ep_rewards = []
    ep_mse = []
    for ep in range(episodes):
        state, info = env.reset()
        done_any = {aid: False for aid in env.agents}
        total_r = 0.0
        mses = []
        while not any(done_any.values()):
            cont_actions, _ = maddpg.get_action(state, training=False)
            state, reward, termination, truncation, info = env.step(cont_actions)
            done_any = {aid: termination[aid] or truncation[aid] for aid in env.agents}
            total_r += float(np.mean(list(reward.values())))
            # the env stores mse in info for each agent; take mean
            mses.append(np.mean([v["mse"] for v in info.values()]))
        ep_rewards.append(total_r)
        ep_mse.append(float(np.mean(mses)))
        print(f"Episode {ep}: return={total_r:.3f}, mean MSE={ep_mse[-1]:.3f}")

    print(f"Avg return: {np.mean(ep_rewards):.3f}  Avg MSE: {np.mean(ep_mse):.3f}")


if __name__ == "__main__":
    evaluate()