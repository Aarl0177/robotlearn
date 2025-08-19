# env_drone_formation.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

@dataclass
class FormationConfig:
    n_agents: int = 3
    dt: float = 0.05
    a_max: float = 2.0           # max accel (m/s^2)
    v_max: float = 3.0           # speed limit (m/s)
    world_radius: float = 10.0   # boundary for soft penalty
    max_steps: int = 1000
    d_min: float = 0.5           # min separation (m)

    # Reward weights
    w_pos: float = 2.0
    w_coll: float = 5.0
    w_ctrl: float = 0.05
    w_bound: float = 0.01

    # Bonus when formation MSE < tol for K consecutive steps
    mse_tol: float = 0.05
    bonus: float = 1.0
    bonus_window: int = 20

    # Observation switches
    include_desired_offset: bool = True
    include_neighbor_relpos: bool = True

class DroneFormationEnv(ParallelEnv):
    metadata = {"name": "DroneFormationEnv-v0", "render_modes": [None, "human"]}

    def __init__(self, cfg: FormationConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = cfg or FormationConfig()
        self.agents = [f"drone_{i}" for i in range(self.cfg.n_agents)]
        self.possible_agents = self.agents[:]
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._step_count = 0
        self._bonus_streak = 0
        # State dicts
        self.pos: Dict[str, np.ndarray] = {}
        self.vel: Dict[str, np.ndarray] = {}
        self.des: Dict[str, np.ndarray] = {}  # desired offset vectors

        # Precompute formation desired offsets around origin (regular polygon)
        theta0 = self._rng.uniform(0, 2 * math.pi)
        self._ring_offsets = [
            2.0 * np.array([math.cos(theta0 + 2 * math.pi * i / self.cfg.n_agents),
                             math.sin(theta0 + 2 * math.pi * i / self.cfg.n_agents)], dtype=np.float32)
            for i in range(self.cfg.n_agents)
        ]

        # Build observation/action spaces (same for all agents)
        obs_dim = 4  # (x,y,vx,vy)
        if self.cfg.include_desired_offset:
            obs_dim += 2
        if self.cfg.include_neighbor_relpos:
            obs_dim += 2 * (self.cfg.n_agents - 1)

        high_obs = np.full((obs_dim,), fill_value=1e3, dtype=np.float32)
        self._observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)
        self._action_space = spaces.Box(low=-self.cfg.a_max, high=self.cfg.a_max, shape=(2,), dtype=np.float32)

    # --- PettingZoo API ---
    def observation_space(self, agent: str):
        return self._observation_space

    def action_space(self, agent: str):
        return self._action_space

    @property
    def num_agents(self):
        return len(self.agents)

    def seed(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def _obs_agent(self, i: int) -> np.ndarray:
        aid = self.agents[i]
        p = self.pos[aid]
        v = self.vel[aid]
        parts = [p, v]
        if self.cfg.include_desired_offset:
            parts.append(self.des[aid])
        if self.cfg.include_neighbor_relpos:
            for j, aj in enumerate(self.agents):
                if j == i:
                    continue
                parts.append(self.pos[aj] - p)
        return np.concatenate(parts, dtype=np.float32)

    def _formation_mse(self) -> float:
        # error = (p_i - desired_offset_i) relative to origin (0,0)
        errs = []
        for i, aid in enumerate(self.agents):
            e = self.pos[aid] - self.des[aid]
            errs.append(np.dot(e, e))
        return float(np.mean(errs))

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        self._step_count = 0
        self._bonus_streak = 0
        # random spawn around annulus [1.5, 3.5] meters from origin
        for i, aid in enumerate(self.agents):
            r = self._rng.uniform(1.5, 3.5)
            ang = self._rng.uniform(0, 2 * math.pi)
            self.pos[aid] = np.array([r * math.cos(ang), r * math.sin(ang)], dtype=np.float32)
            self.vel[aid] = self._rng.normal(0.0, 0.2, size=2).astype(np.float32)
            self.des[aid] = self._ring_offsets[i].astype(np.float32)
        obs = {aid: self._obs_agent(i) for i, aid in enumerate(self.agents)}
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):
        cfg = self.cfg
        self._step_count += 1

        # --- integrate dynamics ---
        for aid in self.agents:
            a = np.asarray(actions[aid], dtype=np.float32)
            a = np.clip(a, -cfg.a_max, cfg.a_max)
            self.vel[aid] = self.vel[aid] + cfg.dt * a
            # clamp speed
            speed = np.linalg.norm(self.vel[aid])
            if speed > cfg.v_max:
                self.vel[aid] = (cfg.v_max / (speed + 1e-8)) * self.vel[aid]
            self.pos[aid] = self.pos[aid] + cfg.dt * self.vel[aid]

        # --- compute rewards ---
        mse = self._formation_mse()
        # collisions (pairwise)
        coll_pen = 0.0
        collided = False
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pi = self.pos[self.agents[i]]
                pj = self.pos[self.agents[j]]
                d = np.linalg.norm(pi - pj)
                if d < cfg.d_min:
                    collided = True
                    coll_pen += (cfg.d_min - d)
        # boundary soft penalty (mean radial overshoot beyond world radius)
        bound_pen = 0.0
        for aid in self.agents:
            r = np.linalg.norm(self.pos[aid])
            if r > cfg.world_radius:
                bound_pen += (r - cfg.world_radius)
        bound_pen /= max(1, self.num_agents)

        mean_ctrl = 0.0
        for aid in self.agents:
            mean_ctrl += float(np.dot(actions[aid], actions[aid]))
        mean_ctrl /= max(1, self.num_agents)

        reward_team = (
            - cfg.w_pos * mse
            - cfg.w_coll * coll_pen
            - cfg.w_ctrl * mean_ctrl
            - cfg.w_bound * bound_pen
        )

        # small bonus if consistently tight formation
        if mse < cfg.mse_tol:
            self._bonus_streak += 1
            if self._bonus_streak >= cfg.bonus_window:
                reward_team += cfg.bonus
                self._bonus_streak = 0
        else:
            self._bonus_streak = 0

        rewards = {aid: reward_team for aid in self.agents}

        # --- termination / truncation ---
        terminations = {aid: bool(collided) for aid in self.agents}
        truncations = {aid: bool(self._step_count >= cfg.max_steps) for aid in self.agents}
        infos = {aid: {"mse": mse} for aid in self.agents}

        obs = {aid: self._obs_agent(i) for i, aid in enumerate(self.agents)}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        pass  # (optional) add matplotlib scatter for visualization

    def close(self):
        pass

# Convenience factory for vectorized wrappers

def make_env(n_agents: int = 3, seed: int | None = None, cfg: FormationConfig | None = None):
    cfg = cfg or FormationConfig(n_agents=n_agents)
    def _f():
        return DroneFormationEnv(cfg=cfg, seed=seed)
    return _f