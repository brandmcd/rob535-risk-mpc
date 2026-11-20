# Final project plan

Written Fri Nov 20, 2026. The proposal timeline put a working risk-budgeted MPC and first synthetic results at today; nothing is built yet. This plan keeps the proposal's two-stage design and cuts stage 2 to the fallback the proposal already named.

## Deadlines

| Date | What |
|---|---|
| Fri Nov 20, 23:55 | Hydra slides due (no penalty until Mon Nov 23, 09:00) |
| Mon Nov 23 | Hydra talk |
| Wed Nov 25 to Fri Nov 27 | Thanksgiving |
| Fri Dec 4 | Poster PDF on Canvas (the syllabus prose allows until Dec 7) |
| Mon Dec 7, 15:00 | Poster session; bring the printed poster |
| Fri Dec 11 | Four-page IEEE report, .tex source, repo link. No late submission. |

## Decisions to make today

- [ ] Stage 2 data: go with Occ3D labels plus synthetic noise unless Anuj already has SurroundOcc running on a GPU. Neither this VPS nor the DASC lab PC is free for it.
- [ ] Risk constraint form. Candidate: per-step union bound, sum over steps of P(collision at step k) <= delta, where P at step k is 1 minus the product of (1 - p_cell) over the cells under the car footprint (Blackmore 2011 style). The team has not agreed on this yet.
- [ ] Map size and resolution (for example 40 m x 20 m at 0.5 m), horizon (for example N = 20 at 0.1 s), car footprint (circles or a rectangle).
- [ ] Repo: put this folder on GitHub (private until submission?) and add Anuj and Brian.

## Week 1: Sat Nov 21 to Tue Nov 24 (stage 1 core)

Brandon (controllers)
- [ ] Baseline MPC: kinematic bicycle, CasADi + IPOPT, track a straight lane, closed-loop drive to a goal.
- [ ] Baseline obstacle cost/constraint on a thresholded map (threshold tau in {0.1, 0.3, 0.5, 0.7}).
- [ ] Smooth occupancy lookup in CasADi (bilinear interpolation of the grid, or a Gaussian blob per occupied cell) so IPOPT gets gradients.
- [ ] Risk-budgeted MPC with the agreed constraint and delta in {0.01, 0.05, 0.1}.
- [ ] Fallback when IPOPT is infeasible: brake to a stop, counted in the metrics.

Anuj (maps)
- [ ] Synthetic road map generator: lane, obstacles placed with known probability p, return both the probability map and a sampled true world.
- [ ] Download a small nuScenes subset plus the matching Occ3D labels to the Hetzner volume.

Brian (experiments)
- [ ] Runner: map seed x method x parameter, closed loop, saves a results table.
- [ ] Metrics: collision (sampled world), time to goal, solve time, infeasible count.
- [ ] Plot: collision rate against time to goal, baseline per tau and risk MPC per delta.

## Week 2: Sat Nov 28 to Thu Dec 3 (results and stage 2)

- [ ] Synthetic sweep, enough seeds that the collision-rate error bars separate delta = 0.01 from 0.05 (several hundred runs per setting). (Brian)
- [ ] Calibration check on synthetic maps: feed the planner miscalibrated maps (probabilities sharpened or flattened), show observed rate vs delta, fix with temperature scaling. This keeps the calibration result even if stage 2 slips. (Brian, Brandon)
- [ ] Occ3D loader: flatten 3D labels to a top-down map in the ego frame, add noise to make probabilities. (Anuj)
- [ ] Run both planners on a handful of nuScenes scenes; count collisions against the labels. (all)
- [ ] Poster draft by Wed Dec 2. (all)

## Week 3: Fri Dec 4 to Fri Dec 11

- [ ] Poster PDF submitted Fri Dec 4.
- [ ] Print the poster; Mon Dec 7 session.
- [ ] Report on Overleaf: each person writes their section (proposal task split); intro, related work and results by Sat Dec 9; final pass Thu Dec 10.
- [ ] Repo cleanup: README with how to reproduce the main plot, repo link in the report.

## Cut list, in order, if time runs short

1. nuScenes stage 2 (keep synthetic plus the synthetic calibration study).
2. Temperature scaling (report the calibration mismatch only).
3. Multiple baseline thresholds (keep tau = 0.5 only).

## What Claude can do

- Write the baseline and risk-budgeted MPC on synthetic maps (Brandon's part) for Brandon to review.
- Write the synthetic map generator and runner if Anuj or Brian are behind.
- Draft the poster layout and the report skeleton on Overleaf.
- Read the papers in `papers/` and draft the related-work paragraph.
