# Risk-Budgeted MPC on Learned Occupancy

ROB 535 final project, Fall 2026, Team 37 (Brandon McDonald, Anuj Arora, Brian Demsich).

A kinematic-bicycle MPC (CasADi/IPOPT) that plans on an occupancy probability map and keeps the predicted collision probability of the whole drive under a budget delta. The baseline is the same MPC on a thresholded map (p > tau is an obstacle).

- Stage 1: synthetic roads where the true risk is known exactly.
- Stage 2: nuScenes mini with Occ3D labels and a small LiDAR occupancy network, with temperature scaling for calibration.

Cells within a correlation length (4 m) count as one collision event, so the risk doesn't depend on grid resolution. The budget includes the hazard already driven and a risk-to-go term to the goal, so replanning can't spend it twice. Details are in the module docstrings, starting with `rmpc/maps.py` and `rmpc/mpc.py`.

## Getting started (team)

- [ ] Clone, run Setup, and run `python experiments/plots.py` to check the committed results load
- [ ] Look through `figures/` and `slides/index.html` for where things stand
- [ ] Read `rmpc/mpc.py` and `rmpc/maps.py` (the planner and risk model)
- [ ] Pick an open item from `TODO.md` (failure cases, ablations, poster, report sections) and tell the group

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python casadi numpy scipy matplotlib pandas gdown
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
```

## Stage 1

```bash
python experiments/run_synthetic.py --maps 200 --workers 3   # ~1 h on 3 cores
```

## Stage 2

Download nuScenes mini (LiDAR only) and the Occ3D labels for the 10 mini scenes:

```bash
mkdir -p data/nuscenes data/occ3d && cd data/nuscenes
curl -L https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz | tar xz --exclude='*CAM_*' --exclude='*RADAR_*'
cd ../occ3d
curl -L https://huggingface.co/datasets/RoyYao233/occ3d-nuscenes-transfer/resolve/main/gts.tar.gz | \
  tar xz gts/scene-0061 gts/scene-0103 gts/scene-0553 gts/scene-0655 gts/scene-0757 \
         gts/scene-0796 gts/scene-0916 gts/scene-1077 gts/scene-1094 gts/scene-1100
cd ../..
```

Then:

```bash
python experiments/prep_nusc.py
python experiments/train_occnet.py --epochs 10 --threads 2
python experiments/calib_nusc.py
python experiments/path_calib.py --workers 3
python experiments/run_nusc.py --workers 3
python experiments/plots.py
```

Results go to `results/` and figures to `figures/`.

## Limitations

- Each scenario is one static LiDAR frame, so moving objects are frozen.
- Unobserved cells count as free when scoring collisions.
- The correlation length, margin and blur are hand-set.
- The network uses one LiDAR sweep, not cameras.
