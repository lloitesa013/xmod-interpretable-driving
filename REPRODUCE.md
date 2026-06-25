# Reproducing X-MoD on Bench2Drive

This is **research code** from a single-GPU (RTX 5090, Windows + WSL) CARLA 0.9.15 / Bench2Drive-220
setup. It is the complete, auditable method behind the paper. **Turn-key one-click reproduction is not
possible from this repo alone**: the pipeline depends on the full CARLA `leaderboard` + `scenario_runner`
stack and on a teacher dataset that is *regenerated*, not bundled (it is several GB). Absolute paths and
the conda-env name in the scripts (`/mnt/c/xmod_b2d`, env `lead`, `/home/<user>/miniconda3`) reflect the
author's environment — **adjust them to yours**. What is published here is the full, readable method so
the results can be audited and adapted.

## Code map
| file | role |
|---|---|
| `code/models/x_mod_vla.py` | the X-MoD model: CNN backbone → router (softmax gate over named **safety / legality / comfort / efficiency** experts) → action head. Architectures: `moe`, `separated`, `targetspeed`. The paper's driving result uses **`targetspeed`** (`[steer, target_speed_norm, 0]`, **no learned brake**). `V_MAX_MPS` denormalizes the target speed. |
| `code/xmod_b2d_agent.py` | the Bench2Drive leaderboard agent. Loads a checkpoint, builds the model from the checkpoint's `model_arch`, and at runtime pairs the predicted target speed with a **classical control stack** (PID + safety shield + anti-stall floor). Requires CARLA `leaderboard` + `scenario_runner`. |
| `code/longitudinal_controller.py` | the `LongitudinalPID` (target speed → throttle/brake). |
| `code/train_xmod_m1.py` | the trainer. Imitation on PDM-Lite teacher data with gate-alignment + sparsity + risk auxiliaries. Key flags: `--model-arch targetspeed --ego-mode route_v4 --lambda-moving 0`; route subsetting via `--include-routes` / `--exclude-routes`; `--sampler-mode {route,label,uniform}`. |
| `code/convert_garage_to_xmod.py` | converts carla_garage / PDM-Lite datagen output into the X-MoD per-frame dataset format. |
| `code/scripts/run_xmod_v2_targetspeed_train.sh` | the training wrapper (the paper's `c0` recipe: 43 routes / 8,456 frames, 20 epochs). |
| `code/scripts/run_xmod_m1_eval.sh` | the closed-loop evaluation wrapper. `XMOD_TM_SEED` sets the traffic-manager seed (used for the 3-seed robustness check). |

## Environment
- **CARLA 0.9.15** (`CARLA_ROOT`), **Bench2Drive-220** route XMLs, and the CARLA `leaderboard` + `scenario_runner` stack.
- Python 3, PyTorch (CUDA), numpy, pillow, opencv.
- Teacher: the privileged **PDM-Lite** expert (carla_garage `autopilot.py` with `DATAGEN=1`).

## Pipeline
1. **Generate teacher data** with PDM-Lite (`DATAGEN=1`, `SAVE_PATH=...`) over your route set.
2. **Convert** to X-MoD format: `python code/convert_garage_to_xmod.py ...` → a dataset dir with
   `metadata.jsonl` (per-frame image / state / partial / action / label + `teacher_target_speed`).
3. **Train** (targetspeed): `bash code/scripts/run_xmod_v2_targetspeed_train.sh` — or call
   `train_xmod_m1.py` directly with `--model-arch targetspeed --ego-mode route_v4 --lambda-moving 0
   --epochs 20`.
4. **Evaluate** closed-loop:
   `XMOD_TARGETSPEED=1 XMOD_EGO_MODE=route_v4 bash code/scripts/run_xmod_m1_eval.sh <ROUTE_ID> <CKPT> <TAG>`.

## Reproducing the headline result
The robust milestone — route **24330** (unseen), **100% completion / 0 collisions across 3 traffic
seeds** — corresponds to:
```bash
for SEED in 0 1 2; do
  XMOD_TARGETSPEED=1 XMOD_EGO_MODE=route_v4 XMOD_TM_SEED=$SEED \
    bash code/scripts/run_xmod_m1_eval.sh 24330 <c0_checkpoint>.pt seed$SEED
done
```
The `c0` checkpoint is trained on 43 routes / 8,456 frames; **24330 is not in the training set.**

## Non-claims
This does **not** solve Bench2Drive and does **not** generalize uniformly (other unseen routes reach
only 18–53%); route completion comes from the policy **plus** the classical control stack, not the
policy alone; the decomposition's validated value is **explainability** (gate/safety-event correlation
0.80), not a driving gain. See `README.md` and the paper's Limitations section.
