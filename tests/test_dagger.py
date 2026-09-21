"""Offline DAgger recovery-label regression coverage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from smartpick_vla.data.dagger import DaggerCollectionConfig, collect_dagger_dataset
from smartpick_vla.data.dataset import TrajectoryDataset, load_trajectory_npz


@dataclass
class _ZeroDecision:
    action: np.ndarray


class _ZeroController:
    def reset(self) -> None:
        return None

    def act(self, observation: dict[str, object]) -> _ZeroDecision:
        del observation
        return _ZeroDecision(np.zeros(6, dtype=np.float32))


@pytest.mark.mujoco
def test_dagger_retains_truthful_failed_rollout_with_expert_labels(tmp_path: Path) -> None:
    destination = tmp_path / "dagger.npz"
    manifest = collect_dagger_dataset(
        destination,
        controller=_ZeroController(),
        config=DaggerCollectionConfig(
            episodes=1,
            seed=701,
            image_size=32,
            max_episode_steps=8,
            mission_length=1,
            expert_action_probability=0.0,
            instruction_splits=("paraphrase",),
            ood_layout_probability=0.0,
        ),
    )
    arrays = load_trajectory_npz(destination)
    dataset = TrajectoryDataset(
        destination,
        action_horizon=4,
        observation_horizon=3,
        successful_only=False,
    )

    assert manifest["attempted_episodes"] == 1
    assert manifest["successful_behavior_episodes"] == 0
    assert manifest["disclosure"]["physical_robot_data"] is False
    assert arrays["action"].shape[1] == 6
    assert arrays["episode_success"].tolist() == [False] * 8
    assert dataset[0]["action"].shape == (4, 6)
    assert dataset[0]["history_mask"].tolist() == [False, False, True]


def test_dagger_config_rejects_unsafe_contract_changes() -> None:
    with pytest.raises(ValueError, match="six-axis"):
        DaggerCollectionConfig(six_axis=False)
    with pytest.raises(ValueError, match="instruction_splits"):
        DaggerCollectionConfig(instruction_splits=("unsupported",))
