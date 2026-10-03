import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tep import ControlMode, TEPSimulator


class Env(gym.Env):
    """
    Base Gym envs for TEP.
    - ControlMode: MANUAL
    - Action: absolute MV values (0..100) for selected mv_indices
    - Observation: XMEAS (41,)
    - Reward: 0.0 (to be shaped by wrapper or external logic)
    """

    def __init__(self, mv_indices=(10,), rl_dt_seconds=60, max_steps=300, freeze_other_mvs=True):
        super().__init__()
        self.mv_indices = tuple(mv_indices)           # 1-based
        self.mv_idx0 = [i - 1 for i in self.mv_indices]
        self.rl_dt_seconds = int(rl_dt_seconds)
        self.max_steps = int(max_steps)
        self.freeze_other_mvs = bool(freeze_other_mvs)

        self.sim = None
        self._step = 0
        self._mv = None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=100.0, shape=(len(self.mv_indices),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = TEPSimulator(control_mode=ControlMode.MANUAL, random_seed=seed)
        self.sim.initialize()

        mv0 = self.sim.get_manipulated_vars().astype(np.float32)
        self._mv = mv0.copy()
        self._step = 0

        obs = self.sim.get_measurements().astype(np.float32)
        info = {"mv": self._mv.copy()}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == len(self.mv_idx0)

        # set selected MVs to absolute values
        for k, idx0 in enumerate(self.mv_idx0):
            self._mv[idx0] = float(np.clip(action[k], 0.0, 100.0))

        # apply MVs
        if self.freeze_other_mvs:
            for i in range(12):
                self.sim.set_mv(i + 1, float(self._mv[i]))
        else:
            for k, idx0 in enumerate(self.mv_idx0):
                self.sim.set_mv(idx0 + 1, float(self._mv[idx0]))

        ok = self.sim.step(n_steps=self.rl_dt_seconds)
        shutdown = self.sim.is_shutdown()

        obs = self.sim.get_measurements().astype(np.float32)

        self._step += 1
        terminated = (not ok) or shutdown
        truncated = self._step >= self.max_steps

        reward = 0.0  # reward shaping is done outside (wrapper)

        info = {
            "shutdown": shutdown,
            "mv": self._mv.copy(),
            "step": self._step,
        }
        return obs, reward, terminated, truncated, info
