#!/usr/bin/env bash
# Report what data exists locally and on Google Drive.
set -uo pipefail
source "$(dirname "$0")/common.sh"
repo="$(cd "$(dirname "$0")/../.." && pwd)"

echo "== Local disk"
df -h "$repo" | tail -1
echo
echo "== Local runtime data (var/)"
du -sh "$repo"/var/* 2>/dev/null
echo
echo "== Google Drive (${SHOP_DRIVE})"
if [[ -x "$RCLONE" ]] && "$RCLONE" listremotes | grep -qx "${SHOP_DRIVE_REMOTE}:"; then
  "$RCLONE" about "${SHOP_DRIVE_REMOTE}:" 2>/dev/null | sed 's/^/  /'
  for top in raw/merl_shopping raw/own annotations splits manifests derived/features checkpoints; do
    printf '  %-12s ' "$top"
    "$RCLONE" size "${SHOP_DRIVE}/${top}" 2>/dev/null | tr '\n' ' '
    echo
  done
else
  echo "  not configured (run: $RCLONE config create ${SHOP_DRIVE_REMOTE} drive scope=drive.file)"
fi
