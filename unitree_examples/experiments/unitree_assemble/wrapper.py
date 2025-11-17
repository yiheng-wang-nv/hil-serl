from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym

from unitree_examples.interventions import UnitreeXRIntervention
from serl_robot_infra.unitree_env import UnitreeG1DirectEnv, UnitreeVisionWrapper


@dataclass
class UnitreeAssembleEnvConfig:
    simulation: bool = True
    action_dt: float = 0.02
    use_safety: bool = True
    use_vision: bool = True


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
