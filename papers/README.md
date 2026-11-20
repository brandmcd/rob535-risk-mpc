# Papers

Run `./fetch.sh` to download the PDFs. They are gitignored.

| File | Paper | What we take from it |
|---|---|---|
| surroundocc_2023.pdf | Wei et al., SurroundOcc, ICCV 2023 | The perception network: six camera images to a 3D occupancy grid with per-voxel probabilities. Stage 2 input. |
| splanning_2024.pdf | Michaux, Isaacson et al., Let's Make a Splan, arXiv 2409.16915 | The risk constraint: bound the total collision probability of a planned trajectory by a budget. We apply the idea to an occupancy grid. |
| occ3d_2023.pdf | Tian et al., Occ3D, NeurIPS D&B 2023 | Ground-truth occupancy labels for nuScenes. Used to count real collisions, and as the base map for the labels-plus-noise fallback. |
| nuscenes_2020.pdf | Caesar et al., nuScenes, CVPR 2020 | The driving dataset (scenes, ego poses, camera images). |
| guo_calibration_2017.pdf | Guo et al., On Calibration of Modern Neural Networks, ICML 2017 | Temperature scaling and reliability diagrams, for the calibration check. |

Related work for the report (no PDF here):

- Blackmore, Ono, Williams, "Chance-constrained optimal path planning with obstacles," IEEE T-RO 27(6), 2011. Risk allocation across time steps with a union bound; the standard citation for splitting a total budget into per-step budgets.
- Zhu and Alonso-Mora, "Chance-constrained collision avoidance for MAVs in dynamic environments," IEEE RA-L 4(2), 2019. Chance-constrained nonlinear MPC turned into a deterministic constraint.
