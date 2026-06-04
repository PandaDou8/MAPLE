#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="configs/quickstart_visualization.yaml"

echo "[MAPLE] Quickstart config: ${CONFIG}"

# Uncomment the next line if you want to run MAPLE training first.
# python script/train.py -c configs/train.yaml

python script/visualize_disease_microbes.py -c "${CONFIG}"
