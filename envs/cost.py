import numpy as np


class Cost:
    """
    TEP economic cost ($/h) adapted to this envs using XMEAS directly.

    Uses:
      - XMEAS 19: Stripper Steam Flow
      - XMEAS 20: Compressor Work
      - XMEAS 10: Purge Rate (stream 9)
      - XMEAS 29,31~36: Purge gas components A, C..H (mole %)
      - XMEAS 37~39: Product components D,E,F (mole %)
      - XMEAS 17: Stripper Underflow (stream 11) -> product flow (for product value term)
    """

    def __init__(
        self,
        w_steam=0.0318,
        w_comp=0.0536,
        purge_scale=0.044791,
        purge_prices=(2.209, 6.177, 22.06, 14.56, 17.89, 30.44, 22.94),  # A,C,D,E,F,G,H
        product_prices=(0.2206, 0.1456, 0.1789),  # D,E,F
        product_flow_index=17,  # 1-based XMEAS index used for "product value"
    ):
        self.w_steam = float(w_steam)
        self.w_comp = float(w_comp)
        self.purge_scale = float(purge_scale)
        self.purge_prices = np.asarray(purge_prices, dtype=np.float64)       # (7,)
        self.product_prices = np.asarray(product_prices, dtype=np.float64)   # (3,)

        self.idx_steam = 19 - 1
        self.idx_comp = 20 - 1
        self.idx_purge_rate = 10 - 1

        # purge comps: A, C, D, E, F, G, H  (B 제외)
        self.idx_purge = np.array([29, 31, 32, 33, 34, 35, 36], dtype=np.int64) - 1

        # product comps: D, E, F
        self.idx_prod_comp = np.array([37, 38, 39], dtype=np.int64) - 1

        self.idx_prod_flow = int(product_flow_index) - 1  # default: XMEAS 17

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float64)

        # energy
        steam = float(obs[self.idx_steam])
        comp = float(obs[self.idx_comp])
        energy_cost = self.w_steam * steam + self.w_comp * comp

        # purge loss
        purge_rate = float(obs[self.idx_purge_rate])
        purge_comps = obs[self.idx_purge]  # (7,)
        purge_cost = (self.purge_scale * purge_rate) * float(purge_comps @ self.purge_prices)

        # product value (reduces cost)
        prod_flow = float(obs[self.idx_prod_flow])
        prod_comps = obs[self.idx_prod_comp]  # (3,)
        product_value = prod_flow * float(prod_comps @ self.product_prices)

        # total cost
        total_cost = energy_cost + purge_cost - product_value
        return float(total_cost)