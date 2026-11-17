# Shared settings for Shop data scripts. Source this file; do not execute it.
# Override with environment variables, e.g. SHOP_DRIVE_REMOTE=mydrive ./drive_setup.sh
SHOP_DRIVE_REMOTE="${SHOP_DRIVE_REMOTE:-shopdrive}"
SHOP_DRIVE_ROOT="${SHOP_DRIVE_ROOT:-ShopliftingAI-Data}"
SHOP_DRIVE="${SHOP_DRIVE_REMOTE}:${SHOP_DRIVE_ROOT}"
RCLONE="${RCLONE:-$(command -v rclone || echo "$HOME/.local/bin/rclone")}"

# Layout matches the data actually held (MERL Shopping) plus the own-footage slot the guides require.
SHOP_DATA_DIRS=(
  raw/merl_shopping/videos raw/merl_shopping/labels raw/own
  annotations splits manifests derived/features evaluation checkpoints
)

require_remote() {
  if [[ ! -x "$RCLONE" ]]; then
    echo "rclone not found. Install it to ~/.local/bin (see docs/data-collection.md)." >&2
    exit 1
  fi
  if ! "$RCLONE" listremotes | grep -qx "${SHOP_DRIVE_REMOTE}:"; then
    echo "rclone remote '${SHOP_DRIVE_REMOTE}:' is not configured." >&2
    echo "Run: $RCLONE config create ${SHOP_DRIVE_REMOTE} drive scope=drive.file" >&2
    exit 1
  fi
}
