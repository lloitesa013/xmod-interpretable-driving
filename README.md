# X-MoD: Interpretable Mixture-of-Drives on Bench2Drive

**A self-driving policy you can read.** It routes every decision through four named drives —
*safety, legality, comfort, efficiency* — so you can see *why* it brakes or goes, tested
**closed-loop** in CARLA / Bench2Drive-220, with the limits reported honestly.

**Result at a glance** — on an **unseen** route (24330, not in the training set):

| seed | route completion | collisions | lane departures |
|:----:|:----------------:|:----------:|:---------------:|
|  0   |     **100%**     |   **0**    |      **0**      |
|  1   |     **100%**     |   **0**    |      **0**      |
|  2   |     **100%**     |   **0**    |      **0**      |

…and the *safety* drive's activation tracks real safety events at correlation **0.80**.

**The honest catch:** it does **not** generalize uniformly — other unseen routes reach only 18–53%,
and this repo shows exactly which fixes (more data, failure-correction data) fail, and why.
Bounded, not hyped.

**Run the headline result** (needs CARLA 0.9.15 + Bench2Drive — see [`REPRODUCE.md`](REPRODUCE.md)):
```bash
for SEED in 0 1 2; do
  XMOD_TARGETSPEED=1 XMOD_EGO_MODE=route_v4 XMOD_TM_SEED=$SEED \
    bash code/scripts/run_xmod_m1_eval.sh 24330 <c0_checkpoint>.pt seed$SEED
done
# -> RouteCompletion 100 %  |  CollisionTest 0 times  |  OutsideRouteLanes 0 %   (x3)
```

## Claims (honestly bounded)
- **Explainability (validated).** The router's *safety* expert activation correlates **0.80** with
  safety events (event 0.65 vs non-event 0.04) at ~1.5 ms/step — a transparent, measurable account of
  *why* the policy acts.
- **The over-brake is the action space, not the decomposition.** Under direct pedal regression the
  interpretable model sits on a safety/mobility seesaw (route completion 3–5%) that survives data
  scale, loss balancing, width, and a structural fork. Removing the model's **learned brake output**
  (it then cannot over-brake; braking delegated to a standard safety shield) breaks the seesaw on the
  *identical* model and data.
- **It drives — on some unseen routes.** Paired with a classical control stack (PID on the predicted
  target speed + safety shield + anti-stall floor), route **24330 (unseen) completes 100% / 0
  collisions / 0 lane departures, robust across 3 traffic seeds**.

## NON-claims (read these)
- **It does NOT solve Bench2Drive and does NOT drive in general.** Of five evaluation routes only one
  (24211) is in training; on the four *unseen* routes the policy completes one at 100% but reaches
  only 18–53% on the others, with collisions. The gap is **scenario-dependent**, not a clean
  train/test split (the lone trained route still collides; the cleanest route is unseen).
- **The residual is not a cheap data fix.** Quadrupling the training routes left *mean* unseen
  completion flat (≈51% → ≈49%); adding privileged-expert failure-correction data **lowered** held-out
  completion and degraded steering (even class-balanced), a naive mix collapsing the policy as the
  safety-heavy failure states reintroduced over-caution. Points to capacity + a *carefully retained*
  on-policy loop, not a quick patch.
- **Completion comes from the policy PLUS a standard control stack** (PID + safety/anti-stall shields),
  not the policy alone. The decomposition is **not** shown to *help* driving; its validated value is
  *explainability*. Results are single-checkpoint (the 24330 milestone is three-seed); the explanation
  result is primarily offline.

## Results — unseen generalization (target-speed head + classical control stack)
| route | in training? | route completion % (collisions) |
|------:|:------------:|:-------------------------------|
| 24330 | unseen       | **100 (0)** — robust across 3 traffic seeds |
| 24240 | unseen       | 53 (1) — blocked behind a scenario obstacle |
| 24781 | unseen       | 25 (1) |
| 24841 | unseen       | 18 (2) |
| 24211 | **trained**  | 88 (1) |

## Paper
`paper/main.pdf` — *The Action Space, Not the Decomposition: Why an Interpretable Mixture-of-Drives
Over-Brakes, and What Makes It Drive.* (6 pages.)

## Code & reproducibility
The complete method is in [`code/`](code/) — the model ([`models/x_mod_vla.py`](code/models/x_mod_vla.py)),
the Bench2Drive leaderboard agent ([`xmod_b2d_agent.py`](code/xmod_b2d_agent.py)), the longitudinal PID,
the trainer ([`train_xmod_m1.py`](code/train_xmod_m1.py)), the dataset converter, and the train / eval
wrappers ([`scripts/`](code/scripts/)). See [`REPRODUCE.md`](REPRODUCE.md) for the pipeline (PDM-Lite
teacher → convert → train → closed-loop eval) and the exact command for the 3-seed 24330 result.

This is research code from a single-GPU CARLA 0.9.15 / Bench2Drive-220 setup: it is the full, auditable
method, but turn-key one-click reproduction needs the CARLA `leaderboard` + `scenario_runner` stack and
a regenerated teacher dataset (not bundled, ~GB). Paths in the scripts reflect the author's environment
— adjust to yours. It is published consistent with the reproducibility bar of the companion
[Bench2Drive failure-taxonomy](https://github.com/lloitesa013/bench2drive-failure-taxonomy) repo. Issues welcome.

## Citation
See `CITATION.cff`.

## License
MIT
