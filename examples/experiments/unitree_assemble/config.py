from __future__ import annotations

from experiments.config import DefaultTrainingConfig
from .wrapper import UnitreeAssembleEnvConfig, make_unitree_assemble_env


class UnitreeAssembleConfig(DefaultTrainingConfig):
    """Minimal configuration placeholder for Unitree assemble tasks."""

    image_keys = ["video.room_view", "video.left_wrist_view", "video.right_wrist_view"]
    proprio_keys = ["robot_state"]
    setup_mode = "dual-arm-learned-gripper"
    max_traj_length = 1000

    def __init__(self, *, simulation: bool = True) -> None:
        self._simulation = simulation

    def get_environment(
        self,
        fake_env: bool = False,
        save_video: bool = False,
        classifier: bool = False,
    ):
        env_cfg = UnitreeAssembleEnvConfig(
            simulation=self._simulation,
            use_safety=not fake_env,
            use_vision=True,
        )
        return make_unitree_assemble_env(env_cfg)

    def process_demos(self, demo):
        # Placeholder: real processing will be defined once the data format is finalised.
        return demo
