import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActionWrapper(gym.Wrapper):
    """
    - Normalized action in [-1, 1]
    - Converts to delta-MV updates and passes absolute MV values to base envs
      (base envs expects action = selected MV values in [0, 100]).

    - Reward is computed by reward_fn(obs, prev_mv, mv, shutdown=...)
      where reward_fn can be any callable (e.g., Reward class instance).

    - Optional: call reward_fn.reset() at episode reset (useful for per-episode scaling).
    """

    def __init__(self, env: gym.Env, delta_mv_max=2.0, reward_fn=None, reset_reward=False):
        super().__init__(env)
        self.delta_mv_max = float(delta_mv_max)
        self.reward_fn = reward_fn
        self.reset_reward = bool(reset_reward)

        n = int(np.prod(env.action_space.shape))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

        self._last_mv = None
        self._prev_mv = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # base envs returns full MV vector in info["mv"]
        mv_full = info.get("mv", None)
        if mv_full is None:
            raise KeyError("Base envs reset() must return info['mv'] as full MV vector (shape (12,)).")

        self._last_mv = np.asarray(mv_full, dtype=np.float32).copy()
        self._prev_mv = self._last_mv.copy()

        # optional reward reset (per-episode scaling)
        if self.reset_reward and (self.reward_fn is not None) and hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # delta MV in [-delta_mv_max, +delta_mv_max]
        delta = np.clip(action, -1.0, 1.0) * self.delta_mv_max

        self._prev_mv = self._last_mv.copy()
        mv_full = self._last_mv.copy()

        mv_idx0 = self.env.mv_idx0  # 0-based indices for selected MVs (from base envs)
        if len(mv_idx0) != delta.shape[0]:
            raise ValueError(f"Action dim mismatch: got {delta.shape[0]} but envs controls {len(mv_idx0)} MVs.")

        # apply delta updates to selected MVs (absolute MV kept in [0, 100])
        for k, idx0 in enumerate(mv_idx0):
            mv_full[idx0] = float(np.clip(mv_full[idx0] + float(delta[k]), 0.0, 100.0))

        # base envs expects only selected MV values
        abs_action = mv_full[mv_idx0].astype(np.float32)

        obs, base_reward, terminated, truncated, info = self.env.step(abs_action)

        # update last MV from envs info
        mv_full_next = info.get("mv", None)
        if mv_full_next is None:
            raise KeyError("Base envs step() must return info['mv'] as full MV vector (shape (12,)).")
        self._last_mv = np.asarray(mv_full_next, dtype=np.float32).copy()

        shutdown = bool(info.get("shutdown", False))
        # ---- add raw metrics to info for logging ----
        # production (raw): from obs
        # (reward_fn이 어떤 생산량 index를 쓰는지 모를 수 있으니, reward_fn에 prod_idx0가 있으면 그걸 사용)
        if self.reward_fn is not None and hasattr(self.reward_fn, "prod_idx0"):
            info["raw_prod"] = float(obs[self.reward_fn.prod_idx0])
        else:
            info["raw_prod"] = None

        # cost (raw): from reward_fn.cost_fn(obs) if available
        if self.reward_fn is not None and hasattr(self.reward_fn, "cost_fn"):
            info["raw_cost"] = float(self.reward_fn.cost_fn(obs))
        else:
            info["raw_cost"] = None
        # compute reward
        if self.reward_fn is None:
            reward = float(base_reward)
        else:
            reward = float(self.reward_fn(obs, self._prev_mv, self._last_mv, shutdown=shutdown))

        return obs, reward, terminated, truncated, info