#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="configs/quickstart_visualization.yaml"

echo "[MAPLE] Quickstart config: ${CONFIG}"

# Full training is intentionally disabled for the quickstart.
# Uncomment the next line if you want to run MAPLE training first.
# python script/train.py -c configs/对抗6层-logger.yaml

python script/visualize_disease_microbes.py -c "${CONFIG}"
