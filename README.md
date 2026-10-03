# Privacy-Aware Multi-Agent Reinforcement Learning on the Tennessee Eastman Process

I studied cooperative control of the Tennessee Eastman Process (TEP) with three agents that observe different parts of the plant. This portfolio contains a split PPO architecture with bottleneck information (BI), selective federated averaging, privacy regularization, matched reinforcement-learning baselines, and diagnostics of what happens when the *value path* receives global information. It is a focused code and evidence selection from a larger research project; the historical checkpoints and operations system are outside this repository.

## 1. Research Question

- Can three agents coordinate process control using only their local measurements at action time?
- Which actor and critic components should share parameters, and what changes when privacy losses act on transmitted BI representations?
- How do the split method and independent, centralized-critic, and fully centralized PPO controls compare under matched evaluation contracts?
- Does giving the BI/value path access to global measurements change the observed learning trajectory? What can a single training seed actually establish?

## 2. Environment

TEP is supplied by the external [`tep` simulator](https://huggingface.co/spaces/jkitchin/tennessee-eastman-process), version **2.5.0**, reference commit `48db70303787e883ae45244801e7699fb991a609`. The reported GlobalBI evaluation used its **Python backend on CPU**. The simulator source is not vendored here; see [setup and provenance](docs/REPRODUCIBILITY.md).

| Contract | Implementation |
|---|---|
| Base observation | 41 XMEAS measurements from [`Env`](envs/env.py) |
| Agents | 3, with local observation sizes **7 / 10 / 8** ([configuration](scripts/train_fl_split_shallow_hybrid.py)) |
| Controlled variables | One action per agent, mapped to 1-based manipulated-variable indices **1 / 8 / 6** |
| Action | Policy output in `[-1, 1]`; [`ActionWrapper`](envs/wrappers.py) scales it to an MV change and clips absolute MVs to `[0, 100]` |
| Time | At most **300** RL steps per episode; each RL step advances **60 simulator seconds** |
| Normalization | Separate running mean and variance for each local observation slice; normalized values are clipped to `[-10, 10]` |

The project's reward uses a *profit proxy*, not an audited monetary profit: `profit = XMEAS[17] - Cost(XMEAS)` and `reward = (profit - baseline) / scale`, with defaults `baseline = -11` and `scale = 5` ([reward](envs/reward.py), [cost](envs/cost.py)). `Cost` combines energy and purge costs with a product-value credit. The episode terminates when the simulator step fails or the plant shuts down; reaching step 300 is a separate time-limit truncation. The wrapper records raw production and cost alongside reward.

## 3. Architecture

```text
local observation (7 / 10 / 8) ──→ local actor ──→ one local action per agent
                 │
                 └─→ local observation + previous local action
                       └─→ local BI encoder / GRU ──→ optional bottleneck ──→ three-node GAT
                                                        └─→ server value estimates
```

I implemented the [local actor](models/fl_split_actor.py), [BI encoder and GAT value model](models/fl_split_critic.py), [optional bottleneck](models/fl_split_bottleneck.py), [PPO trainer](trainers/fl_split_ppo_trainer.py), and [parameter averaging](utils/fedavg.py). The bottleneck module is present in the architecture but is enabled only when configured. PPO computes advantages with GAE, updates actors with a clipped policy ratio and entropy term, and trains the server value estimates against returns.

FedAvg is selective. `fedavg_scope` chooses the actor observation enhancer, the critic GRU, both, or neither. The privacy experiments also have a separate BI-extractor averaging schedule for the GRU and `f_B`; the server GAT and every parameter of every network are **not** averaged together. See [`run_fedavg`](trainers/fl_split_ppo_trainer.py) and the [privacy experiment configuration](scripts/experiment_fl_fedavg10_privacy_full.py).

## 4. Privacy Experiments

The [privacy module](models/bi_privacy_defense.py) implements gradient reversal for adversarial reconstruction (AR), a reconstruction-based negative regularizer (NR), and distance correlation (dCor). The [full privacy trainer](trainers/fl_split_ppo_trainer_privacy_full.py) first trains a reconstructor on detached BI features, then uses the updated reconstructor in the BI privacy update. Its reconstruction target is the **local normalized observation plus previous local action**, including in the F/M/G diagnostic.

The earlier `module_only`/v3 path is a **detached-sidecar diagnostic**: coefficient changes there did not establish that the main BI learned a private representation. The [full experiment entry point](scripts/experiment_fl_fedavg10_privacy_full.py) connects privacy gradients to the BI path. Post-hoc R² and combined/componentwise results have more complex source staging and are not presented here as a headline privacy-performance claim. The [post-hoc probe implementation](scripts/evaluate_privacy_coef_posthoc_r2.py) is included for inspection.

## 5. Matched RL Baselines

The repository includes independent local-critic [IPPO](trainers/ippo_ppo_trainer.py), non-sharing [MAPPO/HAPPO](trainers/noshare_mappo_happo_trainer.py) with a centralized recurrent critic, and [Fully Centralized PPO](trainers/fully_centralized_ppo_trainer.py) with a joint actor and critic. Actor architecture and parameter counts were checked as comparison controls; they do not make the methods identical. The fully centralized experiments distinguish a width-64 common-backbone control from a width-120 parameter-matched capacity control. The four-way CSV below contains **Ours, IPPO, MAPPO, and HAPPO only**.

The following values come directly from [`final_utility_method_aggregate.csv`](results/baselines/final_utility_method_aggregate.csv): three **training** seeds (42/43/44), 1,000 evaluation episodes per training seed, and mean ± sample SD **across training-seed means**. Return and profit are episode sums in the project's reward/profit-proxy units.

| Method | Profit mean ± SD | Return mean | Shutdown episodes | Mean episode length |
|---|---:|---:|---:|---:|
| Split + FedAvg (Ours) | 26.51 ± 32.85 | 163.83 | 100% | 72.06 |
| MAPPO | −137.94 ± 0.68 | 127.79 | 100% | 70.63 |
| HAPPO | −139.26 ± 1.17 | 128.51 | 100% | 71.07 |
| IPPO | −139.92 ± 0.66 | 129.34 | 100% | 71.51 |

These aggregate results show a utility difference under that evaluation contract. **Every method also has a 100% episode shutdown fraction in this CSV**, so the table does not show sustained safe operation.

## 6. GlobalBI F / M / G Diagnostic

The [F/M/G contract](trainers/fl_split_ppo_trainer_globalbi_phase4.py) changes **BI/value inputs** while all three actors still act on their own local observations:

| Variant | BI input |
|---|---|
| **F — Fresh/local** | Its own normalized local slice (7, 10, or 8 values). |
| **M — Masked joint** | A 25D joint layout made from the three *already normalized* local slices; every foreign-observation position is exactly `float32` zero. |
| **G — Global joint** | The full normalized 25D joint layout for each BI encoder. |

For matched initialization, a fresh F model is created first. Its matching actor, server, sidecar, and BI parameters are copied to M and G. The widened BI input projection copies each agent's own-observation columns and its previous-action column; added foreign-observation columns start at zero and remain **trainable**. The [collector](scripts/train_a0_globalbi_phase4.py) and [full logger](scripts/train_a0_globalbi_full_logger.py) preserve the training contract.

![F/M/G training curves](figures/training_curves.png)

![F/M/G held-out episode distributions](figures/heldout_distributions.png)

The final held-out table below uses the actual values in [`heldout_summary_with_variance.csv`](results/globalbi/heldout_summary_with_variance.csv). The plotted episode-level points are available in [`heldout_distributions_underlying.csv`](results/globalbi/heldout_distributions_underlying.csv). Each policy was trained with **seed 42 only** and evaluated on the same seeds **2000–2999**.

| Variant | Mean return sum | Mean profit-proxy sum | Mean length | Plant shutdowns |
|---|---:|---:|---:|---:|
| F | 212.62 | 79.16 | 89.45 | 1,000 / 1,000 |
| M | 197.90 | 58.62 | 84.63 | 1,000 / 1,000 |
| G | 251.33 | 140.07 | 101.51 | 1,000 / 1,000 |

## 7. Critical Result Limitation

**All 3,000 GlobalBI held-out episodes ended in plant shutdown; none reached the full 300-step horizon.** There is one training seed per F/M/G policy and 1,000 shared evaluation seeds per policy. G had higher episode return, profit-proxy sum, and length under this current-source common evaluation, but this does **not** establish improved safety, a causal benefit of global observation, or success at stable 300-step control. A longer episode can accumulate a larger sum before shutdown. The evaluated actors themselves still use local observations.

## 8. Value-Path Diagnostic

[`g2_central_scalar_ablation.py`](trainers/g2_central_scalar_ablation.py) is a **reference implementation** for an exploratory comparison that retains the three local actors while replacing the G BI/GAT three-value path with a centralized recurrent scalar-value path. The comparison changes how value targets and advantages are formed; it is not a Fully Centralized Actor baseline. The historical G2 worker, queue, checkpoints, and raw results are not part of this portfolio, so this file alone is not a runnable G2 experiment.

For a file-by-file entry point guide, dependency setup, and the boundary between inspectable code and historical-run reproduction, see [Reproducibility notes](docs/REPRODUCIBILITY.md).
