# Final project plan

Updated Fri Nov 20, 2026. Built today: both planners, stage 1, the nuScenes/Occ3D stage 2 pipeline with a CPU occupancy network, and the cell- and path-level calibration checks. See README.md for the method and commands.

## Deadlines

| Date | What |
|---|---|
| Mon Nov 23 | Hydra talk (slides open until 09:00) |
| Fri Dec 4 | Poster PDF on Canvas |
| Mon Dec 7, 15:00 | Poster session; bring the printed poster |
| Fri Dec 11 | Four-page IEEE report, .tex source, repo link. No late submission. |

## Done

- [x] Repo, papers, references
- [x] Baseline MPC (distance-field constraint on p > tau) and risk-budgeted MPC (correlation-length hazard, budget accounting, risk-to-go)
- [x] Stage 1: object-level synthetic worlds, 200-map sweep, miscalibration and temperature-scaling correction
- [x] Stage 2 data: nuScenes mini LiDAR + Occ3D labels, 123 usable scenarios
- [x] Stage 2 network: LiDAR BEV U-Net, 5-fold by scene, per-fold temperature
- [x] Cell-level calibration check (reliability, ECE) before and after temperature scaling
- [x] Path-level calibration check (random rollouts) and fitted path scale
- [x] Stage 2 closed loop: raw / temperature-scaled / path-scaled vs threshold baseline

## Next

- [ ] Look at the stage 2 failure cases (collisions and stops) and pick poster examples
- [ ] Great Lakes: run SurroundOcc on nuScenes mini, write `data/stage2/pred.npz` in the same format, rerun calib_nusc, path_calib, run_nusc, plots
- [ ] Ablation: correlation length / margin / blur on stage 1 (the predicted risk is conservative there)
- [ ] Optional: Platt scaling (temperature plus bias); the network's error is partly a bias
- [ ] Poster draft (figures in `figures/`), then the report on Overleaf
- [ ] Team: hand Anuj and Brian their report sections; add them to the repo

## Cut list if time runs short

1. SurroundOcc on Great Lakes (keep the LiDAR network as the stage 2 source)
2. Ablations
