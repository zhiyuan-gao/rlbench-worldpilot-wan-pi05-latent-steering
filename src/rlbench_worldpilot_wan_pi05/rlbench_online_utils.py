from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


VIEW_TO_ATTR = {
    "front": "front_rgb",
    "left_shoulder": "left_shoulder_rgb",
    "right_shoulder": "right_shoulder_rgb",
}


def install_rlbench_root(rlbench_root: str | Path | None) -> None:
    if rlbench_root is None or not str(rlbench_root):
        return
    root = Path(rlbench_root)
    if root.exists() and root.as_posix() not in sys.path:
        sys.path.insert(0, root.as_posix())


def task_class_from_name(task_name: str):
    module = importlib.import_module(f"rlbench.tasks.{task_name}")
    class_name = "".join(part[:1].upper() + part[1:] for part in task_name.split("_"))
    return getattr(module, class_name)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz = q[..., :3]
    w = np.clip(q[..., 3], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.full_like(angle, 2.0)
    np.divide(angle, sin_half, out=scale, where=sin_half > 1e-8)
    return xyz * scale[..., None]


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.full_like(angle, 0.5)
    np.divide(np.sin(half), angle, out=scale, where=angle > 1e-8)
    xyz = rotvec * scale
    w = np.cos(half)
    return normalize_quat(np.concatenate([xyz, w], axis=-1))


def gripper_open_value(obs, threshold: float = 0.95) -> float:
    joint_positions = getattr(obs, "gripper_joint_positions", None)
    if joint_positions is None:
        return float(getattr(obs, "gripper_open", 0.0))
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    if joint_positions.size == 0:
        return float(getattr(obs, "gripper_open", 0.0))
    return 1.0 if float(joint_positions[0]) / 0.04 > float(threshold) else 0.0


def obs_to_state(obs, *, gripper_threshold: float = 0.95) -> np.ndarray:
    ee_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    if ee_pose.shape != (7,):
        raise ValueError(f"Expected obs.gripper_pose shape (7,), got {ee_pose.shape}")
    return np.concatenate(
        [
            ee_pose[:3],
            quat_to_rotvec(ee_pose[3:7]).astype(np.float32),
            [gripper_open_value(obs, threshold=gripper_threshold)],
        ]
    ).astype(np.float32)


def obs_rgb_images(obs, views: tuple[str, ...] = ("front", "left_shoulder", "right_shoulder")) -> dict[str, np.ndarray]:
    images = {}
    for view in views:
        image = getattr(obs, VIEW_TO_ATTR[view])
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            if np.nanmax(image) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        images[view] = np.ascontiguousarray(image[..., :3])
    return images


def policy_obs_from_rlbench(obs, *, prompt: str, gripper_threshold: float = 0.95) -> dict:
    images = obs_rgb_images(obs)
    return {
        "front_image": images["front"],
        "left_shoulder_image": images["left_shoulder"],
        "right_shoulder_image": images["right_shoulder"],
        "state": obs_to_state(obs, gripper_threshold=gripper_threshold),
        "prompt": prompt,
    }


def read_goal_images_from_episode(
    rgb_episode_dir: str | Path,
    frame_idx: int,
    *,
    views: tuple[str, ...] = ("front", "left_shoulder", "right_shoulder"),
) -> dict[str, np.ndarray]:
    from PIL import Image

    root = Path(rgb_episode_dir)
    images = {}
    for view in views:
        path = root / f"{view}_rgb" / f"{int(frame_idx)}.png"
        images[view] = np.asarray(Image.open(path).convert("RGB"))
    return images


def load_demo(lowdim_episode_dir: str | Path):
    with (Path(lowdim_episode_dir) / "low_dim_obs.pkl").open("rb") as f:
        return pickle.load(f)


def resolve_episode_dir(row: Mapping, root: str | Path) -> Path:
    relpath = row.get("rgb_episode_relpath")
    if relpath:
        return Path(root) / str(relpath)
    return Path(root) / str(row["task"]) / str(row["variation"]) / "episodes" / str(row["episode"])


def make_obs_config(*, image_size: int = 256):
    from rlbench.observation_config import CameraConfig, ObservationConfig

    def camera_config(enabled: bool):
        return CameraConfig(
            rgb=enabled,
            depth=False,
            point_cloud=False,
            mask=False,
            image_size=(image_size, image_size),
        )

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.front_camera = camera_config(True)
    obs_config.left_shoulder_camera = camera_config(True)
    obs_config.right_shoulder_camera = camera_config(True)
    obs_config.overhead_camera = camera_config(False)
    obs_config.wrist_camera = camera_config(False)
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.gripper_joint_positions = True
    return obs_config


def make_action_mode(*, gripper_open_threshold: float = 0.95, stop_on_success: bool = True):
    from pyrep.const import ObjectType
    from pyrep.errors import ConfigurationPathError
    try:
        from pyrep.const import Algos
    except ImportError:
        from pyrep.const import ConfigurationPathAlgorithms as Algos
    from rlbench.action_modes.action_mode import ActionMode
    from rlbench.action_modes.arm_action_modes import (
        EndEffectorPoseViaPlanning,
        RelativeFrame,
        assert_action_shape,
        assert_unit_quaternion,
        calculate_delta_pose,
    )
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.exceptions import InvalidActionError

    class EndEffectorPoseViaPlanningWithIgnoreCollision(EndEffectorPoseViaPlanning):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._stop_on_success = bool(stop_on_success)

        def action(self, scene, action: np.ndarray, ignore_collisions: bool = True):
            assert_action_shape(action, (7,))
            assert_unit_quaternion(action[3:])
            if not self._absolute_mode and self._frame != RelativeFrame.EE:
                action = calculate_delta_pose(scene.robot, action)
            relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
            self._quick_boundary_check(scene, action)

            colliding_shapes = []
            if not ignore_collisions:
                if self._robot_shapes is None:
                    self._robot_shapes = scene.robot.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
                if scene.robot.arm.check_arm_collision():
                    grasped_objects = scene.robot.gripper.get_grasped_objects()
                    colliding_shapes = [
                        shape
                        for shape in scene.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE)
                        if (
                            shape.is_collidable()
                            and shape not in self._robot_shapes
                            and shape not in grasped_objects
                            and scene.robot.arm.check_arm_collision(shape)
                        )
                    ]
                    [shape.set_collidable(False) for shape in colliding_shapes]

            try:
                try:
                    path = scene.robot.arm.get_path(
                        action[:3],
                        quaternion=action[3:],
                        ignore_collisions=bool(ignore_collisions),
                        relative_to=relative_to,
                        trials=100,
                        max_configs=10,
                        max_time_ms=10,
                        trials_per_goal=5,
                        algorithm=Algos.RRTConnect,
                    )
                except ConfigurationPathError as exc:
                    if ignore_collisions:
                        raise InvalidActionError(
                            "A path could not be found. Most likely due to the target being inaccessible "
                            "or a collision was detected."
                        ) from exc
                    path = scene.robot.arm.get_path(
                        action[:3],
                        quaternion=action[3:],
                        ignore_collisions=True,
                        relative_to=relative_to,
                        trials=100,
                        max_configs=10,
                        max_time_ms=10,
                        trials_per_goal=5,
                        algorithm=Algos.RRTConnect,
                    )
            finally:
                [shape.set_collidable(True) for shape in colliding_shapes]

            done = False
            while not done:
                done = path.step()
                scene.step()
                if self._stop_on_success:
                    success, _ = scene.task.success()
                    if success:
                        break

    class MoveArmThenGripperWithIgnoreCollision(ActionMode):
        """Action layout: [abs_xyz, quat_xyzw, gripper_action, ignore_collision]."""

        def action(self, scene, action: np.ndarray):
            arm_act_size = int(np.prod(self.arm_action_mode.action_shape(scene)))
            arm_action = np.asarray(action[:arm_act_size], dtype=np.float64)
            gripper_action = np.asarray(action[arm_act_size : arm_act_size + 1], dtype=np.float64)
            ignore_collision_action = np.asarray(action[arm_act_size + 1 : arm_act_size + 2], dtype=np.float64)
            if ignore_collision_action.shape != (1,):
                raise InvalidActionError(f"Expected ignore_collision shape (1,), got {ignore_collision_action.shape}")
            self.arm_action_mode.action(scene, arm_action, ignore_collisions=bool(ignore_collision_action[0] > 0.5))
            self.gripper_action_mode.action(scene, gripper_action)

        def action_shape(self, scene):
            return (
                int(np.prod(self.arm_action_mode.action_shape(scene)))
                + int(np.prod(self.gripper_action_mode.action_shape(scene)))
                + 1
            )

    class DiscreteWithOpenThreshold(Discrete):
        def __init__(self, open_threshold: float = 0.95):
            super().__init__(attach_grasped_objects=True, detach_before_open=True)
            self._open_threshold = float(open_threshold)

        def action(self, scene, action: np.ndarray):
            assert_action_shape(action, self.action_shape(scene.robot))
            if not 0.0 <= float(action[0]) <= 1.0:
                raise InvalidActionError("Gripper action expected to be within 0 and 1.")

            open_condition = all(x > self._open_threshold for x in scene.robot.gripper.get_open_amount())
            current_ee = 1.0 if open_condition else 0.0
            action_value = float(action[0] > 0.5)

            if current_ee != action_value:
                if not self._detach_before_open:
                    self._actuate(action_value, scene)
                if action_value == 0.0 and self._attach_grasped_objects:
                    for graspable in scene.task.get_graspable_objects():
                        scene.robot.gripper.grasp(graspable)
                else:
                    scene.robot.gripper.release()
                if self._detach_before_open:
                    self._actuate(action_value, scene)
                if action_value == 1.0:
                    for _ in range(10):
                        scene.pyrep.step()
                        scene.task.step()

    return MoveArmThenGripperWithIgnoreCollision(
        EndEffectorPoseViaPlanningWithIgnoreCollision(
            absolute_mode=True,
            collision_checking=False,
        ),
        DiscreteWithOpenThreshold(gripper_open_threshold),
    )


def action7_to_rlbench_action9(action7: np.ndarray, *, ignore_collision: bool = True) -> np.ndarray:
    action7 = np.asarray(action7, dtype=np.float64)
    quat = rotvec_to_quat(action7[3:6]).astype(np.float64)
    gripper = np.clip(float(action7[6]), 0.0, 1.0)
    return np.concatenate(
        [
            action7[:3],
            quat,
            [gripper, 1.0 if ignore_collision else 0.0],
        ]
    ).astype(np.float64)


def pose_event_reached(
    obs,
    goal_obs,
    *,
    pos_threshold: float,
    rot_threshold: float,
    gripper_threshold: float,
) -> tuple[bool, dict[str, float]]:
    obs_pose = np.asarray(obs.gripper_pose, dtype=np.float64)
    goal_pose = np.asarray(goal_obs.gripper_pose, dtype=np.float64)
    pos_error = float(np.linalg.norm(obs_pose[:3] - goal_pose[:3]))
    rot_error = float(np.linalg.norm(quat_to_rotvec(obs_pose[3:7]) - quat_to_rotvec(goal_pose[3:7])))
    grip_error = float(abs(gripper_open_value(obs) - gripper_open_value(goal_obs)))
    reached = pos_error <= pos_threshold and rot_error <= rot_threshold and grip_error <= gripper_threshold
    return reached, {
        "event_goal_pos_error": pos_error,
        "event_goal_rotvec_error": rot_error,
        "event_goal_gripper_error": grip_error,
    }
