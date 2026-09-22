"""History-aware action-chunk VLA for partially observed manipulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn

from smartpick_vla.models.encoders import ByteTextEncoder, CompactVisionEncoder, count_parameters


@dataclass(frozen=True, slots=True)
class TemporalVLAConfig:
    """Configuration for a compact history-aware vision-language-action policy.

    The model consumes a left-padded fixed-length observation history. Each
    history item contains one camera image and the matching robot state. The
    policy is intentionally compact enough for a laptop-scale smoke run while
    retaining the temporal-token and action-query structure used by larger
    embodied policies.
    """

    robot_state_dim: int = 24
    action_dim: int = 5
    action_horizon: int = 8
    observation_horizon: int = 4
    d_model: int = 128
    nhead: int = 4
    temporal_layers: int = 2
    decoder_layers: int = 2
    language_layers: int = 1
    feedforward_dim: int = 256
    language_max_length: int = 64
    vision_grid_size: int = 4
    # ``mean`` preserves the original checkpoint contract. ``flatten`` keeps
    # the learned CoordConv grid's spatial layout before temporal fusion and
    # is preferred for new grounding-focused training runs.
    vision_pooling: str = "mean"
    dropout: float = 0.0
    squash_actions: bool = True
    freeze_vision: bool = False
    freeze_language: bool = False
    # Auxiliary heads are trained only when the dataset carries synthetic
    # target-pixel/stage labels.  Keeping them in the model config makes the
    # checkpoint self-describing while preserving the original action API.
    stage_classes: int = 9

    def __post_init__(self) -> None:
        positive = (
            self.robot_state_dim,
            self.action_dim,
            self.action_horizon,
            self.observation_horizon,
            self.d_model,
            self.nhead,
            self.temporal_layers,
            self.decoder_layers,
            self.language_layers,
            self.feedforward_dim,
            self.language_max_length,
            self.vision_grid_size,
            self.stage_classes,
        )
        if any(value < 1 for value in positive):
            raise ValueError("TemporalVLAConfig dimensions and layer counts must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.vision_pooling not in {"mean", "flatten"}:
            raise ValueError("vision_pooling must be 'mean' or 'flatten'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class TemporalVLAPolicy(nn.Module):
    """Fuse temporal vision/proprioception tokens before action-chunk decoding.

    Every history step is encoded into one visual token and one robot-state
    token. A Transformer encoder fuses those pairs across time, then a second
    Transformer decoder attends over temporal and language tokens to predict a
    short continuous action chunk. ``history_mask`` carries true values for
    valid observations and supports episode-safe left padding.
    """

    requires_observation_history = True

    def __init__(self, config: TemporalVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = CompactVisionEncoder(
            config.d_model, grid_size=config.vision_grid_size
        )
        self.visual_projection = (
            nn.Linear(config.d_model * config.vision_grid_size * config.vision_grid_size, config.d_model)
            if config.vision_pooling == "flatten"
            else nn.Identity()
        )
        self.language_encoder = ByteTextEncoder(
            config.d_model,
            max_length=config.language_max_length,
            num_layers=config.language_layers,
            nhead=config.nhead,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(config.robot_state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.time_embedding = nn.Embedding(config.observation_horizon, config.d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.temporal_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.action_queries = nn.Embedding(config.action_horizon, config.d_model)
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.action_dim),
        )
        self.goal_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, 2),
        )
        self.stage_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.stage_classes),
        )
        if config.freeze_vision:
            _freeze(self.vision_encoder)
        if config.freeze_language:
            _freeze(self.language_encoder)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self.config.freeze_vision:
            self.vision_encoder.eval()
        if self.config.freeze_language:
            self.language_encoder.eval()
        return self

    def forward(
        self,
        rgb_history: Tensor,
        instruction: Sequence[str] | Tensor,
        robot_state_history: Tensor,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict an action chunk from a fixed-length observation history."""

        actions, _ = self.forward_with_aux(
            rgb_history,
            instruction,
            robot_state_history,
            history_mask,
        )
        return actions

    def forward_with_aux(
        self,
        rgb_history: Tensor,
        instruction: Sequence[str] | Tensor,
        robot_state_history: Tensor,
        history_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return actions plus optional spatial/phase predictions.

        ``goal_xy`` is normalized image coordinates in ``[0, 1]`` and
        ``stage_logits`` follows the eight waypoint stages plus ``done``.
        The heads are deliberately auxiliary: callers that only need the
        historical action contract can continue using :meth:`forward`.
        """

        if rgb_history.ndim != 5:
            raise ValueError("rgb_history must have shape [B,T,C,H,W]")
        if robot_state_history.ndim != 3:
            raise ValueError("robot_state_history must have shape [B,T,D]")
        batch_size, history_steps, channels, height, width = rgb_history.shape
        if channels != 3:
            raise ValueError("rgb_history must contain three RGB channels")
        if history_steps != self.config.observation_horizon:
            raise ValueError(
                "rgb_history length must match observation_horizon "
                f"({self.config.observation_horizon}), got {history_steps}"
            )
        if robot_state_history.shape != (
            batch_size,
            history_steps,
            self.config.robot_state_dim,
        ):
            raise ValueError(
                "robot_state_history must have shape "
                f"[B,{history_steps},{self.config.robot_state_dim}]"
            )
        if history_mask is None:
            history_mask = torch.ones(
                (batch_size, history_steps), dtype=torch.bool, device=rgb_history.device
            )
        if history_mask.shape != (batch_size, history_steps):
            raise ValueError("history_mask must have shape [B,T]")
        if history_mask.dtype is not torch.bool:
            raise TypeError("history_mask must use bool dtype")
        if not bool(history_mask.any(dim=1).all()):
            raise ValueError("every batch item must contain at least one valid observation")

        flattened_rgb = rgb_history.reshape(batch_size * history_steps, channels, height, width)
        visual_grid = self.vision_encoder(flattened_rgb)
        if self.config.vision_pooling == "flatten":
            # Preserve row/column identity. The grid already carries learned
            # spatial embeddings; flattening makes the location recoverable by
            # the projection instead of cancelling it with a global mean.
            visual_tokens = self.visual_projection(visual_grid.reshape(batch_size * history_steps, -1))
            visual_tokens = visual_tokens.reshape(batch_size, history_steps, -1)
        else:
            visual_tokens = visual_grid.mean(dim=1).reshape(batch_size, history_steps, -1)
        state_tokens = self.state_encoder(
            robot_state_history.to(dtype=visual_tokens.dtype, device=visual_tokens.device)
        )
        time_indices = torch.arange(history_steps, device=visual_tokens.device)
        temporal_input = (
            visual_tokens + state_tokens + self.time_embedding(time_indices).unsqueeze(0)
        )
        temporal_tokens = self.temporal_encoder(
            temporal_input,
            src_key_padding_mask=~history_mask.to(device=visual_tokens.device),
        )

        language_tokens, language_padding = self.language_encoder(instruction)
        if language_tokens.shape[0] != batch_size:
            raise ValueError("rgb_history and instruction batch sizes must match")
        memory = torch.cat((temporal_tokens, language_tokens), dim=1)
        memory_padding = torch.cat(
            (
                ~history_mask.to(device=memory.device),
                language_padding.to(device=memory.device),
            ),
            dim=1,
        )
        queries = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.action_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_padding,
        )
        action_chunk = self.action_head(decoded)
        actions = torch.tanh(action_chunk) if self.config.squash_actions else action_chunk
        latest_index = history_steps - 1
        latest_token = temporal_tokens[:, latest_index]
        aux = {
            "goal_xy": torch.sigmoid(self.goal_head(latest_token)),
            "stage_logits": self.stage_head(latest_token),
        }
        return actions, aux

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)

    @torch.no_grad()
    def act(
        self,
        rgb_history: Tensor,
        instruction: Sequence[str] | Tensor,
        robot_state_history: Tensor,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the first executable action from the predicted action chunk."""

        return self(rgb_history, instruction, robot_state_history, history_mask)[:, 0]
