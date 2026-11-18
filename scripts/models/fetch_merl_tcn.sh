#!/usr/bin/env bash
# Fetch the MERL shelf-action research checkpoint from Drive and verify it against its manifest.
# Usage: scripts/models/fetch_merl_tcn.sh            (needs the rclone remote from docs/data-collection.md)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE="${SHOP_DRIVE_REMOTE:-shopdrive}:${SHOP_DRIVE_ROOT:-ShopliftingAI-Data}"
RCLONE="${RCLONE:-$(command -v rclone || echo "$HOME/.local/bin/rclone")}"
RUN="tcn-20261117T221342Z"
DEST="$ROOT/var/models/merl_tcn_v1"
mkdir -p "$DEST"
"$RCLONE" copy "$REMOTE/checkpoints/merl_tcn_v1/$RUN" "$DEST" 2> >(grep -v "shared Google Drive client_id" >&2)
EXPECTED="$(python3 -c "import json;print(json.load(open('$ROOT/ml/manifests/merl_action_tcn_dev.json'))['weights']['sha256'])")"
ACTUAL="$(sha256sum "$DEST/model.pt" | cut -d' ' -f1)"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "SHA-256 mismatch for model.pt: expected $EXPECTED got $ACTUAL" >&2
  rm -f "$DEST/model.pt"
  exit 1
fi
echo "MERL TCN checkpoint verified: $DEST/model.pt ($ACTUAL)"
