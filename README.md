# X-MoD: Interpretable Mixture-of-Drives on Bench2Drive

An **interpretable** driving policy — a transparent Mixture-of-Drives whose safety gate tracks real
risk — evaluated **closed-loop** on Bench2Drive-220 (CARLA 0.9.15), with the privileged PDM-Lite
expert as teacher. This repository releases the paper and an **honest, precisely-bounded** account of
what the interpretable decomposition does and does not buy for driving.

> One line: the gate explains risk faithfully (event/non-event correlation **0.80**); with its learned
> brake output removed and a classical control stack, the model completes an **unseen** route at
> **100% / 0 collisions, robust across three traffic seeds** — but it does **not** generalize
> uniformly, and we show exactly which fixes fail and why.

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
The model, the Bench2Drive leaderboard agent, and the evaluation harness are **being packaged for
reproducible release** (the pipeline is CARLA 0.9.15 / Bench2Drive-220 specific). This will land as a
follow-up so that what is published actually runs — consistent with the reproducibility bar of the
companion [Bench2Drive failure-taxonomy](https://github.com/lloitesa013/bench2drive-failure-taxonomy)
repository. Until then, please open an issue for specifics.

## Citation
See `CITATION.cff`.

## License
MIT
