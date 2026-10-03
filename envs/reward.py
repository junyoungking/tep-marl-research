import numpy as np


class Reward:
    """
    Reward = (profit - baseline) / scale
           = (production - cost - baseline) / scale

    - baseline, scale are fixed constants
    - objective is preserved as long as scale > 0
    """

    def __init__(
        self,
        cost_fn,
        production_index=17,   # 1-based XMEAS index
        baseline=-11.0,
        scale=5.0,
        eps=1e-8,
    ):
        self.cost_fn = cost_fn
        self.prod_idx0 = int(production_index) - 1
        self.baseline = float(baseline)
        self.scale = float(scale)
        self.eps = float(eps)

    def reset(self):
        pass

    def __call__(self, obs, prev_mv, mv, shutdown=False):
        obs = np.asarray(obs, dtype=np.float64)

        p = float(obs[self.prod_idx0])
        c = float(self.cost_fn(obs))
        profit = p - c

        reward = (profit - self.baseline) / max(self.scale, self.eps)
        return float(reward)