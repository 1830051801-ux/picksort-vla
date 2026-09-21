"""Offline DAgger collection for recovery-oriented VLA imitation learning.

The collector runs entirely inside :class:`~smartpick_vla.envs.smartpick_env.SmartPickEnv`.
At each state visited by a mixed learner/expert behaviour policy, the privileged
IK expert supplies an action label.  The resulting archive intentionally keeps
the *actual* episode success flag separate from label eligibility: an expert
label remains useful when the learner has drifted to a recoverable state.

This module never opens a real robot transport.  It is a Sim2Real preparation
tool, not a hardware data collector.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import torch

from smartpick_vla.data.dataset import save_trajectory_npz, validate_trajectory_arrays
from smartpick_vla.data.expert import IKWaypointExpert
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.envs.smartpick_env import SmartPickEnv
from smartpick_vla.utils.io import atomic_write_json
from smartpick_vla.utils.provenance import sha256_file


class _ControllerDecision(Protocol):
    action: np.ndarray


class DaggerController(Protocol):
    """Small controller surface required by the offline collector."""

    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any]) -> _ControllerDecision | np.ndarray: ...


@dataclass(frozen=True, slots=True)
class DaggerCollectionConfig:
    """Deterministic configuration for mixed-policy offline re-labelling."""

    episodes: int = 48
    seed: int = 20260921
    image_size: int = 64
    max_episode_steps: int = 380
    grasp_assist: bool = True
    six_axis: bool = True
    mission_length: int = 3
    expert_action_probability: float = 0.40
    instruction_splits: tuple[str, ...] = ("train", "paraphrase", "ood")
    ood_layout_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if self.image_size < 32 or self.image_size > 192:
            raise ValueError("image_size must be in [32,192]")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if self.mission_length < 1 or self.mission_length > 3:
            raise ValueError("mission_length must be in [1,3]")
        if not self.six_axis:
            raise ValueError("DAgger collection currently targets the six-axis action contract")
        if not 0.0 <= self.expert_action_probability <= 1.0:
            raise ValueError("expert_action_probability must be in [0,1]")
        if not 0.0 <= self.ood_layout_probability <= 1.0:
            raise ValueError("ood_layout_probability must be in [0,1]")
        splits = tuple(self.instruction_splits)
        if not splits or any(split not in {"train", "paraphrase", "ood"} for split in splits):
            raise ValueError("instruction_splits must contain train, paraphrase, and/or ood")
        object.__setattr__(self, "instruction_splits", splits)


def _normalized_policy_action(value: _ControllerDecision | np.ndarray, action_dim: int) -> np.ndarray:
    """Extract a safe normalized action without depending on controller classes."""

    candidate = getattr(value, "action", value)
    action = np.asarray(candidate, dtype=np.float32)
    if action.shape != (action_dim,) or not np.isfinite(action).all():
        raise ValueError(f"policy controller returned an invalid action shape {action.shape}")
    return np.clip(action, -1.0, 1.0)


def collect_dagger_dataset(
    output_path: str | Path,
    *,
    controller: DaggerController,
    config: DaggerCollectionConfig,
    domain_config: DomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    """Collect learner-visited, privileged-expert-labelled synthetic transitions.

    ``episode_success`` describes the mixed behaviour rollout truthfully.  The
    archive deliberately retains failed attempts because every stored state has
    a finite expert recovery label; callers must opt in explicitly before
    training on failed-attempt labels.
    """

    randomization = domain_config or DomainRandomizationConfig(enabled=True)
    environment = SmartPickEnv(
        image_size=config.image_size,
        max_episode_steps=config.max_episode_steps,
        domain_randomization=randomization,
        grasp_assist=config.grasp_assist,
        six_axis=config.six_axis,
        mission_length=config.mission_length,
    )
    rng = np.random.default_rng(config.seed)
    records: dict[str, list[Any]] = {
        "rgb": [],
        "robot_state": [],
        "action": [],
        "episode_id": [],
        "step_index": [],
        "instruction": [],
        "task_class": [],
        "episode_success": [],
        "goal_xy": [],
        "goal_visible": [],
        "stage_index": [],
    }
    buffers: list[dict[str, list[Any]]] = []
    episode_summaries: list[dict[str, Any]] = []
    try:
        for episode_id in range(config.episodes):
            split = config.instruction_splits[episode_id % len(config.instruction_splits)]
            ood_layout = bool(rng.random() < config.ood_layout_probability)
            observation, _ = environment.reset(
                seed=config.seed + episode_id,
                options={"instruction_split": split, "ood_layout": ood_layout},
            )
            expert = IKWaypointExpert(environment, use_noisy_detection=True)
            expert.reset()
            controller.reset()
            buffer: dict[str, list[Any]] = {key: [] for key in records}
            expert_behavior_steps = 0
            policy_behavior_steps = 0
            action_distances: list[float] = []
            episode_return = 0.0
            info: dict[str, Any] = {}
            for step_index in range(config.max_episode_steps):
                expert_action, diagnostics = expert.act()
                goal_xy, goal_visible = expert.target_pixel_xy()
                policy_action = _normalized_policy_action(
                    controller.act(observation), environment.action_dim
                )
                expert_action = np.asarray(expert_action, dtype=np.float32)
                action_distances.append(float(np.linalg.norm(policy_action - expert_action)))
                use_expert = bool(rng.random() < config.expert_action_probability)
                behaviour_action = expert_action if use_expert else policy_action
                expert_behavior_steps += int(use_expert)
                policy_behavior_steps += int(not use_expert)

                buffer["rgb"].append(np.asarray(observation["rgb"], dtype=np.uint8).copy())
                buffer["robot_state"].append(
                    np.asarray(observation["robot_state"], dtype=np.float32).copy()
                )
                buffer["action"].append(expert_action.copy())
                buffer["episode_id"].append(episode_id)
                buffer["step_index"].append(step_index)
                buffer["instruction"].append(str(observation["instruction"]))
                buffer["task_class"].append(environment.task.target_class)
                buffer["episode_success"].append(False)
                buffer["goal_xy"].append(goal_xy)
                buffer["goal_visible"].append(goal_visible)
                buffer["stage_index"].append(diagnostics.stage_index)

                observation, reward, terminated, truncated, info = environment.step(behaviour_action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            success = bool(info.get("success", False))
            buffer["episode_success"] = [success] * len(buffer["action"])
            buffers.append(buffer)
            episode_summaries.append(
                {
                    "episode_id": episode_id,
                    "seed": config.seed + episode_id,
                    "instruction_split": split,
                    "ood_layout": ood_layout,
                    "success": success,
                    "steps": len(buffer["action"]),
                    "return": episode_return,
                    "collision_steps": int(info.get("collision_steps", 0)),
                    "wrong_pick": bool(info.get("wrong_pick", False)),
                    "wrong_bin": bool(info.get("wrong_bin", False)),
                    "expert_behavior_steps": expert_behavior_steps,
                    "policy_behavior_steps": policy_behavior_steps,
                    "mean_policy_expert_action_l2": (
                        float(np.mean(action_distances)) if action_distances else None
                    ),
                    "randomization": info.get("randomization", {}),
                }
            )
    finally:
        environment.close()

    for buffer in buffers:
        for key in records:
            records[key].extend(buffer[key])
    arrays: dict[str, np.ndarray] = {
        "rgb": np.asarray(records["rgb"], dtype=np.uint8),
        "robot_state": np.asarray(records["robot_state"], dtype=np.float32),
        "action": np.asarray(records["action"], dtype=np.float32),
        "episode_id": np.asarray(records["episode_id"], dtype=np.int32),
        "step_index": np.asarray(records["step_index"], dtype=np.int32),
        "instruction": np.asarray(records["instruction"], dtype=np.str_),
        "task_class": np.asarray(records["task_class"], dtype=np.str_),
        "episode_success": np.asarray(records["episode_success"], dtype=np.bool_),
        "goal_xy": np.asarray(records["goal_xy"], dtype=np.float32),
        "goal_visible": np.asarray(records["goal_visible"], dtype=np.bool_),
        "stage_index": np.asarray(records["stage_index"], dtype=np.int64),
    }
    statistics = validate_trajectory_arrays(arrays)
    destination = save_trajectory_npz(output_path, arrays)
    successful = sum(int(item["success"]) for item in episode_summaries)
    manifest = {
        "schema_version": "smartpick-dagger-demonstrations/v1",
        "dataset_file": destination.name,
        "dataset_sha256": sha256_file(destination),
        "collection": asdict(config),
        "statistics": asdict(statistics),
        "attempted_episodes": config.episodes,
        "successful_behavior_episodes": successful,
        "failed_behavior_episodes_retained": config.episodes - successful,
        "episodes": episode_summaries,
        "disclosure": {
            "offline": True,
            "physical_robot_data": False,
            "state_distribution": "mixed learner/expert MuJoCo rollout",
            "label_source": "privileged IK expert queried on learner-visited simulator states",
            "failed_attempt_labels": "retained; training requires explicit opt-in",
            "grasp_assist": config.grasp_assist,
            "arm_variant": environment.arm_variant,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
        },
    }
    atomic_write_json(destination.with_suffix(".manifest.json"), manifest)
    return manifest


__all__ = ["DaggerCollectionConfig", "DaggerController", "collect_dagger_dataset"]
