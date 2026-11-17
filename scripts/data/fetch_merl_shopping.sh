#!/usr/bin/env bash
# Rebuild raw/merl_shopping/ on Google Drive from the official MERL distribution.
# Only needed for a fresh Drive; the current Drive already holds the dataset.
# Needs ~3.6 GB free local space temporarily (1.7 GB zip + extraction); cleaned up afterwards.
# Licence: research and educational use only (see ReadMe.md "Legalese").
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_remote

base="https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset"
work="$(mktemp -d "${TMPDIR:-/tmp}/merl.XXXXXX")"
trap 'rm -rf "$work"' EXIT

for zip in Labels_MERL_Shopping_Dataset.zip Videos_MERL_Shopping_Dataset.zip; do
  echo "download ${zip}"
  curl -fL --retry 3 -C - -o "${work}/${zip}" "${base}/${zip}"
  unzip -tq "${work}/${zip}"
  unzip -q "${work}/${zip}" -d "${work}/x"
done
curl -fsSL -o "${work}/ReadMe.md" "${base}/ReadMe.md"

dest="${SHOP_DRIVE}/raw/merl_shopping"
find "${work}/x" -name '*_crop.mp4' -exec "$RCLONE" copy {} "${dest}/videos" \;
find "${work}/x" -name '*_label.mat' -exec "$RCLONE" copy {} "${dest}/labels" \;
"$RCLONE" copyto "${work}/ReadMe.md" "${dest}/ReadMe.md"

echo "Rebuilding annotations, split and manifest..."
"$RCLONE" md5sum "$dest" --exclude ReadMe.md > "${work}/md5.txt"
mkdir -p "${work}/labels" && find "${work}/x" -name '*_label.mat' -exec cp {} "${work}/labels/" \;
repo="$(cd "$(dirname "$0")/../.." && pwd)"
"$repo/.venv/bin/python" -I "$repo/scripts/data/prepare_merl.py" "${work}/labels" "${work}/out" --md5 "${work}/md5.txt"
"$RCLONE" copy "${work}/out" "${SHOP_DRIVE}"
