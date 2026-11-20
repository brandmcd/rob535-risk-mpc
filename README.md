# Risk-Budgeted MPC on Learned Occupancy

ROB 535 / NAVARCH 565 final project, Fall 2026, Team 37 (Brandon McDonald, Anuj Arora, Brian Demsich).

A kinematic-bicycle MPC in CasADi that plans directly on an occupancy probability map and keeps the trajectory's total collision probability below a budget delta. We compare it with the same MPC on a thresholded map, first on synthetic maps with known probabilities, then on nuScenes occupancy (SurroundOcc predictions, or Occ3D labels plus noise), and check whether the observed collision rate matches delta before and after temperature scaling.

- Proposal and report: Overleaf project 6aea1be794b5336901e2dc0e
- Papers: `papers/` (run `papers/fetch.sh`), BibTeX in `refs.bib`
- Plan: `TODO.md`

Planned layout:

```
maps/         synthetic maps, nuScenes/Occ3D loading, top-down flattening   (Anuj)
control/      baseline MPC, risk-budgeted MPC                                (Brandon)
experiments/  runner, sweeps, metrics, plots, calibration                    (Brian)
```
