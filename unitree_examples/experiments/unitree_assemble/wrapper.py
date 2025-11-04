from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym

from unitree_examples.interventions import UnitreeXRIntervention
from serl_robot_infra.unitree_env import UnitreeG1DirectEnv, UnitreeVisionWrapper


class EpisodeLimitWrapper(gym.Wrapper):
    """Simple wrapper that truncates an episode after a fixed number of steps."""

    def __init__(self, env: gym.Env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if (
            not terminated
            and not truncated
            and self._elapsed_steps >= self._max_episode_steps
        ):
            truncated = True
            info = dict(info)
            info.setdefault("time_limit_reached", True)
        return obs, reward, terminated, truncated, info


@dataclass
class UnitreeAssembleEnvConfig:
    simulation: bool = True
    action_dt: float = 0.02
    use_safety: bool = False
    use_vision: bool = True
    max_episode_steps: int = 1000


def make_unitree_assemble_env(config: UnitreeAssembleEnvConfig) -> gym.Env:
    """Build the base Unitree assemble environment with optional safety/vision wrappers."""
    base_env = UnitreeG1DirectEnv(
        arm="G1_29",
        ee="dex3",
        simulation=config.simulation,
        action_dt=config.action_dt,
    )

    safety_kwargs = dict(
        go_home_steps=int(max(1, 4.0 / config.action_dt)),
        settle_time=0.5,
        soft_start_duration=5.0,
        health_timeout=0.5,
    )

    env: gym.Env = base_env
    env = UnitreeVisionWrapper(
        env,
        enable_safety=config.use_safety,
        safety_kwargs=safety_kwargs,
    )

    env = UnitreeXRIntervention(env)
    env = EpisodeLimitWrapper(env, max_episode_steps=config.max_episode_steps)

    env = UnitreeAssembleTaskWrapper(env)
    return env


class UnitreeAssembleTaskWrapper(gym.Wrapper):
    """Placeholder task wrapper that injects default reward/info structure."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info.setdefault("succeed", False)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault("succeed", False)
        # placeholder reward; replace with task-specific signal when available
        reward = float(reward)
        return obs, reward, terminated, truncated, info
