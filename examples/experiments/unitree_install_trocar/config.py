import dataclasses
from typing import Dict, Tuple
from types import SimpleNamespace

from examples.experiments.config import DefaultTrainingConfig


@dataclasses.dataclass
class UnitreeRobotParams:
    arm: str = "G1_29"
    ee: str = "dex3"
    simulation: bool = True
    motion_mode: bool = False
    action_dt: float = 0.02


@dataclasses.dataclass
class UnitreeVisionParams:
    server_address: str = "192.168.123.164"
    port: int = 5555
    fps: int = 30
    head_camera_shape: Tuple[int, int] = (480, 640)
    head_camera_id_numbers: Tuple[int, ...] = (4,)
    wrist_camera_shape: Tuple[int, int] = (480, 640)
    wrist_camera_id_numbers: Tuple[int, ...] = (0, 2)
    enable_wrist: bool = True


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
    vision_params: UnitreeVisionParams = UnitreeVisionParams()

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        if fake_env:
            raise NotImplementedError(
                "Offline training uses dataset-derived observation spaces; "
                "unitree_train_bc.py constructs the fake environment directly."
            )

        from gymnasium.wrappers import RecordEpisodeStatistics

        from unitree_env import (
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
        image_config = {
            "fps": self.vision_params.fps,
            "head_camera_type": "opencv",
            "head_camera_image_shape": list(self.vision_params.head_camera_shape),
            "head_camera_id_numbers": list(self.vision_params.head_camera_id_numbers),
        }
        if self.vision_params.enable_wrist:
            image_config.update(
                {
                    "wrist_camera_type": "opencv",
                    "wrist_camera_image_shape": list(self.vision_params.wrist_camera_shape),
                    "wrist_camera_id_numbers": list(self.vision_params.wrist_camera_id_numbers),
                }
            )

        image_args = SimpleNamespace(
            sim=self.unitree_params.simulation,
            image_config=image_config,
            server_address=self.vision_params.server_address,
            port=self.vision_params.port,
        )

        env = UnitreeVisionWrapper(env, enable_safety=False, image_args=image_args)

        if save_video:
            # UnitreeVisionWrapper already exposes images; logging handled elsewhere.
            pass

        env = RecordEpisodeStatistics(env)
        return env
