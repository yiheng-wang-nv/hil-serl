import dataclasses
from typing import Dict

from experiments.config import DefaultTrainingConfig


@dataclasses.dataclass
class UnitreeRobotParams:
    arm: str = "G1_29"
    ee: str = "dex3"
    simulation: bool = True
    motion_mode: bool = False
    action_dt: float = 0.02


class TrainConfig(DefaultTrainingConfig):
    """
    Training configuration for Unitree install trocar task collected with XR teleoperation.
    """

    image_keys = ["video.room_view", "video.left_wrist_view", "video.right_wrist_view"]
    proprio_keys = ["state"]
    classifier_keys = None
    encoder_type = "resnet-pretrained"
    setup_mode = "dual-arm-learned-gripper"

    dataset_state_key: str = "observation.state"
    dataset_action_key: str = "action"
    dataset_camera_map: Dict[str, str] = {
        "observation.images.cam_room": "video.room_view",
        "observation.images.cam_left_wrist": "video.left_wrist_view",
        "observation.images.cam_right_wrist": "video.right_wrist_view",
    }

    unitree_params: UnitreeRobotParams = UnitreeRobotParams()

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        if fake_env:
            raise NotImplementedError(
                "Offline training uses dataset-derived observation spaces; "
                "unitree_train_bc.py constructs the fake environment directly."
            )

        from gymnasium.wrappers import RecordEpisodeStatistics

        from serl_robot_infra.unitree_env import (
            UnitreeG1DirectEnv,
            UnitreeSafetyWrapper,
            UnitreeVisionWrapper,
        )

        env = UnitreeG1DirectEnv(
            arm=self.unitree_params.arm,
            ee=self.unitree_params.ee,
            simulation=self.unitree_params.simulation,
            motion_mode=self.unitree_params.motion_mode,
            action_dt=self.unitree_params.action_dt,
        )
        env = UnitreeSafetyWrapper(env)
        env = UnitreeVisionWrapper(env, enable_safety=False)

        if save_video:
            # UnitreeVisionWrapper already exposes images; logging handled elsewhere.
            pass

        env = RecordEpisodeStatistics(env)
        return env

