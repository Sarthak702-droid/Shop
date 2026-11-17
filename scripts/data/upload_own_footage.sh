#!/usr/bin/env bash
# Move authorised own footage from a local folder (phone/camera dump) to Drive, then free local space.
# Usage: scripts/data/upload_own_footage.sh <local_folder> <session_name>
#   session_name groups recordings for leakage-safe splits, e.g. 2026-11-20_storeA_cam1_actors
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_remote

src="${1:?local folder required}"
session="${2:?session name required, e.g. 2026-11-20_storeA_cam1}"
[[ "$session" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "session name: letters, digits, . _ - only" >&2; exit 1; }

dest="${SHOP_DRIVE}/raw/own/${session}"
echo "Uploading ${src} -> ${dest}"
"$RCLONE" copy --progress --checksum \
  --include '*.{mp4,MP4,mov,MOV,mkv,MKV,webm,WEBM,avi,AVI,m4v,M4V}' "$src" "$dest"
echo "Verifying..."
"$RCLONE" check --one-way --include '*.{mp4,MP4,mov,MOV,mkv,MKV,webm,WEBM,avi,AVI,m4v,M4V}' "$src" "$dest"
echo "Verified. Local copies are still in ${src}; delete them yourself once you are satisfied."
