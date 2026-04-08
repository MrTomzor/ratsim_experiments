"""Custom feature extractors for SB3 policies."""

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LidarCnnExtractor(BaseFeaturesExtractor):
    """Applies a 1D CNN to the 'lidar' observation, concatenates other obs raw.

    The lidar observation is expected to be flat (n_rays * n_channels,).
    It is reshaped internally to (n_channels, n_rays) for Conv1d processing.
    Non-lidar observations (compass, gps, goal, etc.) are concatenated
    directly without transformation.

    n_rays and n_channels are read from the env (env.num_lidar_rays,
    env.num_lidar_channels) at model creation time and stored in
    features_extractor_kwargs so they persist across save/load.

    Args:
        observation_space: Dict observation space (must contain a "lidar" key).
        n_rays: Number of lidar rays (spatial dimension).
        n_channels: Number of channels per ray (1 for depth-only, M+1 for depth+semantics).
        cnn_output_dim: Output dimension of the CNN branch.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        n_rays: int,
        n_channels: int = 1,
        cnn_output_dim: int = 32,
    ):
        self._n_rays = n_rays
        self._n_channels = n_channels

        other_keys = sorted(k for k in observation_space.spaces if k != "lidar")
        other_dim = sum(spaces.flatdim(observation_space[k]) for k in other_keys)
        self._other_keys = other_keys

        super().__init__(observation_space, features_dim=cnn_output_dim + other_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, cnn_output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        lidar = observations["lidar"]
        lidar = lidar.view(lidar.shape[0], self._n_channels, self._n_rays)
        cnn_out = self.cnn(lidar)

        if self._other_keys:
            others = th.cat([observations[k] for k in self._other_keys], dim=-1)
            return th.cat([cnn_out, others], dim=-1)

        return cnn_out
