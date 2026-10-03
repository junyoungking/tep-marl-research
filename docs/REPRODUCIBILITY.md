# Reproducibility and source map

This repository is a small portfolio selection from a larger TEP research workspace. It contains 42 copied research Python files, five evidence files, and five empty package initializers. `envs/env.py` was changed only to import the installed `tep` package instead of inserting a sibling checkout into `sys.path`. The original research workspace and simulator checkout were not modified during assembly.

## Validated environment

The source environment was `pytep_hug` with Python 3.10.19, PyTorch 2.10.0+cpu, NumPy 2.2.6, Gymnasium 1.2.3, PyTorch Geometric 2.7.0, TensorBoard 2.20.0, and `tep` 2.5.0. `requirements.txt` pins the direct Python dependencies using installable distribution version names. Its `torch==2.10.0` line does **not** encode the CPU build suffix; select a CPU PyTorch wheel when matching the validated environment. It also does not freeze all transitive packages.

The TEP simulator is an external dependency: [upstream source](https://huggingface.co/spaces/jkitchin/tennessee-eastman-process), reference commit `48db70303787e883ae45244801e7699fb991a609`. Install that checkout as `tep` separately. The GlobalBI result used the Python backend; set `TEP_BACKEND=python` when matching that contract. The simulator package and its original copyright notices remain upstream.

```bash
python -m pip install -r requirements.txt
python -m pip install -e /path/to/tennessee-eastman-process
TEP_BACKEND=python python -B -c "from tep import TEPSimulator; from envs.env import Env; print('imports OK')"
```

These commands are setup/import guidance. They are not a claim that the historical training was rerun from this portfolio.

## Where to read the implementation

| Question | Entry point and core implementation |
|---|---|
| Split/FedAvg | `scripts/train_fl_split_shallow_hybrid.py` → `trainers/fl_split_ppo_trainer.py` → `models/fl_split_actor.py`, `models/fl_split_critic.py`, `utils/fedavg.py` |
| Privacy | `scripts/experiment_fl_fedavg10_privacy_full.py` → `trainers/fl_split_ppo_trainer_privacy_full.py` → `models/bi_privacy_defense.py` |
| Post-hoc privacy probe | `scripts/evaluate_privacy_coef_posthoc_r2.py` → `scripts/privacy_stage1_utils.py` |
| IPPO | `scripts/train_ippo_architecture_matched.py` → `trainers/ippo_ppo_trainer.py` → `models/ippo_critic.py` |
| MAPPO/HAPPO | `scripts/train_noshare_mappo_happo.py` → `trainers/noshare_mappo_happo_trainer.py` → `models/centralized_recurrent_critic.py` |
| Fully Centralized PPO | `scripts/train_fully_centralized_ppo.py` → `trainers/fully_centralized_ppo_trainer.py` → `models/fully_centralized_actor.py` |
| GlobalBI F/M/G | `scripts/train_a0_globalbi_full_logger.py` → `scripts/train_a0_globalbi_phase4.py` → `trainers/fl_split_ppo_trainer_globalbi_phase4.py` |
| G2 value path | `trainers/g2_central_scalar_ablation.py` is a reference implementation; no G2 launcher is included |

## Scope of the evidence

`results/baselines/final_utility_method_aggregate.csv` summarizes four methods over three training seeds. The GlobalBI images and two CSV files preserve a single-training-seed, 1,000-held-out-episodes-per-policy diagnostic. The GlobalBI held-out plot has episode-level data in `results/globalbi/heldout_distributions_underlying.csv`; the training-curve underlying CSV is not included. All F/M/G held-out episodes ended in shutdown, and no episode completed 300 steps. The baseline aggregate also reports 100% episode shutdown fractions for its four methods.

Historical GlobalBI extension runs used initial bundles, exact-resume checkpoints, a source fingerprint, and an operations journal. They are deliberately absent. The copied `scripts/train_a0_globalbi_phase4.py` still fingerprints backend files in a sibling `tennessee-eastman-process` checkout when its historical run path is invoked. Keeping that guard unmodified preserves the original scientific code but means that importing these modules and replaying the historical extension are different levels of reproducibility. No source code beyond `envs/env.py` was changed for this portfolio.

No `LICENSE` file has been added pending a publication-rights decision for the research code. Check the simulator's own `LICENSE` and copyright notices before publishing this repository.
