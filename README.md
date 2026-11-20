# Risk-Budgeted MPC on Learned Occupancy

ROB 535 / NAVARCH 565 final project, Fall 2026, Team 37 (Brandon McDonald, Anuj Arora, Brian Demsich).

A kinematic-bicycle MPC in CasADi that plans on an occupancy probability map and keeps the predicted collision probability of the whole drive below a budget delta. It is compared with the same MPC on a thresholded map (cells with p > tau are obstacles). Stage 1 uses synthetic roads where the true risk is known exactly. Stage 2 uses nuScenes mini with Occ3D labels, an occupancy network's probabilities, and a calibration check and temperature-scaling correction.

## Method

**Car and dynamics.** Kinematic bicycle with the state at the rear axle (x, y, heading, speed), inputs acceleration and steering, RK4 with dt = 0.2 s, horizon N = 20 (4 s), solved by IPOPT. Speed limit 10 m/s.

**Risk model.** Treating every grid cell as an independent obstacle makes a 30 % blob of 50 cells a near-certain collision, and the answer then depends on the grid resolution. The planner instead treats cells within a correlation length l (4 m, about one object) as one event:

- h(cell) = -log(1 - p)
- M(x) = largest h within half a car width plus 0.3 m of x, blurred by 0.5 m for gradients
- G(x) = M(x) / l, the hazard per meter the car's front edge travels

Driving a path accumulates H = sum G ds, and the predicted collision probability is 1 - exp(-H). The constraint at each solve is

    sum over the horizon of G(front) * ds  +  R(front at horizon end)  <=  -log(1 - delta) - H_spent

where R is the least hazard from the end of the horizon to the goal (Dijkstra), so a plan cannot spend the budget it needs to finish, and H_spent is the hazard already driven, so replanning cannot spend delta twice. If no plan meets the constraint, the car follows the last plan that did, then brakes.

**Baseline.** Same MPC; cells with p > tau are obstacles; three circles covering the car must stay at least their radius (plus 0.2 m) from the nearest obstacle cell, using a distance field.

**Tracking cost.** Both planners follow a Dijkstra cost-to-go on their own map (risk: 1 + 100 G per meter; baseline: 1 per meter, 3 near obstacles, 20 inside), which also gives the warm start. A goal-distance cost left IPOPT stopping in front of obstacles.

**Stage 1 maps** (`rmpc/synthetic.py`). A 60 m road between walls with 4-8 disk obstacles, each existing with probability q in [0.02, 0.6]. The planner sees the per-cell marginal; the true risk of a run is 1 - prod(1 - q) over the obstacles its swept footprint touches, computed exactly. Miscalibrated maps are sigmoid(logit(p) / T); the correction temperature is fitted by maximum likelihood on 50 separate maps with sampled outcomes.

**Stage 2 data** (`rmpc/nusc.py`, `experiments/prep_nusc.py`). nuScenes v1.0-mini (404 keyframes, 10 scenes) and Occ3D-nuScenes labels on the 200 x 200 x 0.4 m ego grid. An obstacle cell is any non-ground class in voxel layers 3-7 (0.2-2.2 m above the road) seen by the LiDAR. Each keyframe gives a scenario: start at the ego pose and speed, goal where the recorded car was 4 s later. Only scenarios where the car moved at least 8 m and its recorded path is free in the labels are used (123 of 404, from 6 scenes).

**Stage 2 network** (`experiments/train_occnet.py`). A CPU stand-in for SurroundOcc: a small U-Net on single-sweep LiDAR bird's-eye-view features (point counts in four height bins, max height, intensity, hit flag), trained with plain cross-entropy on observed cells. Five folds by scene: train on 6 scenes, fit the temperature on 2, predict the held-out 2. Any network that writes `data/stage2/pred.npz` with the same fields can replace it (SurroundOcc on Great Lakes later).

## Reproduce

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python casadi numpy scipy matplotlib pandas gdown
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

# stage 1 (about 1 h on 3 cores)
python experiments/run_synthetic.py --maps 200 --workers 3

# stage 2 data: nuScenes mini (LiDAR + metadata only) and Occ3D labels for the 10 mini scenes
mkdir -p data/nuscenes data/occ3d && cd data/nuscenes
curl -L https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz | tar xz --exclude='*CAM_*' --exclude='*RADAR_*'
cd ../occ3d
curl -L https://huggingface.co/datasets/RoyYao233/occ3d-nuscenes-transfer/resolve/main/gts.tar.gz | \
  tar xz gts/scene-0061 gts/scene-0103 gts/scene-0553 gts/scene-0655 gts/scene-0757 \
         gts/scene-0796 gts/scene-0916 gts/scene-1077 gts/scene-1094 gts/scene-1100
cd ../..

python experiments/prep_nusc.py
python experiments/train_occnet.py --epochs 10 --threads 2
python experiments/calib_nusc.py
python experiments/run_nusc.py --workers 3
python experiments/plots.py
```

Outputs: `results/*.csv`, `results/summary.json`, `results/nusc_calibration.json`, `figures/*.png`.

## Layout

```
rmpc/maps.py         grids, hazard field, distance field, swept footprint
rmpc/mpc.py          CasADi/IPOPT MPC, risk and threshold modes
rmpc/navigation.py   Dijkstra cost-to-go and warm start
rmpc/sim.py          closed-loop episode with budget accounting
rmpc/synthetic.py    stage 1 object-level worlds
rmpc/nusc.py         nuScenes/Occ3D loading, BEV features, scenarios
rmpc/calib.py        temperature scaling, reliability, ECE, Wilson intervals
experiments/         stage 1 and 2 runners, network training, calibration, plots
papers/              cited papers (run papers/fetch.sh)
```

## Known limitations

- Each scenario treats one LiDAR frame as a static world, so moving objects are frozen where they were at the keyframe.
- Unobserved cells count as free when scoring collisions.
- The correlation length l, the 0.3 m margin and the blur are hand-set; they make the predicted risk conservative on stage 1.
- The network sees one LiDAR sweep; SurroundOcc on cameras would give different, likely worse-calibrated, probabilities.
