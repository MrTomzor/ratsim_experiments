"""Bridge a `gymnasium.Env` (WildfireGymEnv) into an `embodied.Env` for DreamerV3.

dreamerv3 ships `embodied.envs.from_gym.FromGym`, but it targets the legacy
`gym` package (4-tuple step, single-value reset). Our env uses gymnasium
(5-tuple step with terminated/truncated, (obs, info) reset). This module
provides a thin adapter that mirrors FromGym for the gymnasium API.
"""
import functools

import elements
import embodied
import gymnasium as gym
import numpy as np


class GymnasiumToEmbodied(embodied.Env):
    """Adapt a gymnasium.Env to the embodied.Env interface used by DreamerV3."""

    def __init__(self, env: gym.Env, obs_key: str = "vector", act_key: str = "action"):
        self._env = env
        self._obs_dict = isinstance(env.observation_space, gym.spaces.Dict)
        self._act_dict = isinstance(env.action_space, gym.spaces.Dict)
        # MultiDiscrete needs per-dim bounds; embodied.Space is single-bound only,
        # so we expose each dim as a separate Discrete key (act_key/0, act_key/1, ...)
        # and recombine on step. nvec is cached for that reconstruction.
        self._md_nvec = (
            np.asarray(env.action_space.nvec, dtype=np.int32)
            if isinstance(env.action_space, gym.spaces.MultiDiscrete)
            else None
        )
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info: dict | None = None

    @property
    def env(self) -> gym.Env:
        return self._env

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        spaces = (
            self._flatten(self._env.observation_space.spaces)
            if self._obs_dict
            else {self._obs_key: self._env.observation_space}
        )
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            # log/ keys are not fed to the agent; embodied.run.train aggregates
            # them per episode (avg, max, sum) and writes to the logger.
            "log/reward_pickups": elements.Space(np.float32),
            "log/distance_traveled": elements.Space(np.float32),
            "log/step_distance": elements.Space(np.float32),
        }

    @functools.cached_property
    def act_space(self):
        if self._md_nvec is not None:
            spaces = {
                f"{self._act_key}_{i}": elements.Space(np.int32, (), 0, int(n))
                for i, n in enumerate(self._md_nvec)
            }
        elif self._act_dict:
            spaces = {k: self._convert(v) for k, v in self._flatten(self._env.action_space.spaces).items()}
        else:
            spaces = {self._act_key: self._convert(self._env.action_space)}
        spaces["reset"] = elements.Space(bool)
        return spaces

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs, self._info = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._md_nvec is not None:
            act = np.array(
                [int(action[f"{self._act_key}_{i}"]) for i in range(len(self._md_nvec))],
                dtype=np.int32,
            )
        elif self._act_dict:
            act = self._unflatten(action)
        else:
            act = action[self._act_key]
        obs, reward, terminated, truncated, self._info = self._env.step(act)
        self._done = bool(terminated or truncated)
        return self._obs(
            obs,
            reward,
            is_last=self._done,
            is_terminal=bool(terminated),
        )

    def render(self):
        return self._env.render()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        # Populate log/ metrics from the underlying gym env.
        env = self._env
        obs["log/reward_pickups"] = np.float32(
            env.get_reward_pickups() if hasattr(env, "get_reward_pickups") else 0
        )
        obs["log/distance_traveled"] = np.float32(
            env.get_distance_traveled() if hasattr(env, "get_distance_traveled") else 0
        )
        obs["log/step_distance"] = np.float32(
            env.longest_step_distance if hasattr(env, "longest_step_distance") else 0
        )
        return obs

    def _flatten(self, nest, prefix=None):
        # Replace '/' with '_' so embodied keys stay flat-friendly.
        result = {}
        for key, value in nest.items():
            full = f"{prefix}/{key}" if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, full))
            else:
                result[full] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return elements.Space(np.int32, (), 0, int(space.n))
        if isinstance(space, gym.spaces.MultiDiscrete):
            return elements.Space(np.int32, space.shape, 0, space.nvec.max())
        return elements.Space(space.dtype, space.shape, space.low, space.high)
