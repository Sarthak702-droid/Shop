#!/usr/bin/env bash
# Create the Shop dataset folder tree on Google Drive (idempotent).
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_remote

echo "Creating ${SHOP_DRIVE}/ ..."
for dir in "${SHOP_DATA_DIRS[@]}"; do
  "$RCLONE" mkdir "${SHOP_DRIVE}/${dir}"
done

readme="$(mktemp)"
cat > "$readme" <<'TXT'
ShopliftingAI dataset storage
=============================
Managed by scripts/data/*.sh in the ShopliftingAI repository. Do not rename folders by hand.

raw/merl_shopping/   MERL Shopping Dataset: 106 overhead videos (41 subjects, 30 fps, 920x680)
  videos/            xx_yy_crop.mp4  (xx = subject, yy = session)
  labels/            xx_yy_label.mat (tlabs: 5 action classes, start/end frames)
  ReadMe.md          Publisher ReadMe incl. licence: research and educational use only
raw/own/             Authorised own-store / consenting-actor footage (future commercial source)
annotations/         Converted labels, e.g. merl_shopping_v1/intervals.csv
splits/              Frozen subject-grouped train / val / test lists
manifests/           Provenance, rights and MD5 per dataset
derived/features/    Cached per-window features for the temporal model
evaluation/          Model outputs and reports
checkpoints/         Trained weights with their manifest
TXT
"$RCLONE" copyto "$readme" "${SHOP_DRIVE}/README.txt"
rm -f "$readme"

echo "Done. Current layout:"
"$RCLONE" lsf -R --dirs-only "${SHOP_DRIVE}"
