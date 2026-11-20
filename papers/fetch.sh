#!/usr/bin/env bash
# Download the papers cited in the proposal. PDFs are not committed (copyright).
set -euo pipefail
cd "$(dirname "$0")"
get(){ [ -s "$1" ] || curl -sSfL -o "$1" "$2"; echo "$1"; }
get surroundocc_2023.pdf     https://arxiv.org/pdf/2303.09551
get splanning_2024.pdf       https://arxiv.org/pdf/2409.16915
get occ3d_2023.pdf           https://arxiv.org/pdf/2304.14365
get nuscenes_2020.pdf        https://arxiv.org/pdf/1903.11027
get guo_calibration_2017.pdf https://arxiv.org/pdf/1706.04599
