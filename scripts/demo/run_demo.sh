#!/usr/bin/env bash
# Reproducible end-to-end demo: upload -> shelf-action job -> wait -> summary -> review link.
# Needs the API (port 8000) and worker running (see docs/runbook.md) and the MERL checkpoint
# (scripts/models/fetch_merl_tcn.sh).
#
#   scripts/demo/run_demo.sh [video] [profile]
#   default video: held-out MERL test video 30_1_crop.mp4 (subject 30, never used in training)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
API="${SHOP_API:-http://127.0.0.1:8000/api/v1}"
VIDEO="${1:-$ROOT/Videos_MERL_Shopping_Dataset/Videos_MERL_Shopping_Dataset/30_1_crop.mp4}"
PROFILE="${2:-merl_shelf_actions_research_v1}"
AUTH=()
[[ -n "${SHOP_API_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $SHOP_API_TOKEN")
py() { python3 -c "import json,sys; d=json.load(sys.stdin); $1"; }

curl -sf "${AUTH[@]}" "$API/health/ready" | py 'print("readiness:", d["status"]); [print("  ", c["name"], c["status"]) for c in d["checks"]]'
echo "uploading $(basename "$VIDEO") ..."
UPLOAD=$(curl -sf "${AUTH[@]}" -F "file=@$VIDEO" "$API/uploads" | py 'print(d["upload"]["upload_id"])')
JOB=$(curl -sf "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "{\"source\":{\"kind\":\"file\",\"upload_id\":\"$UPLOAD\"},\"analysis_profile\":\"$PROFILE\"}" \
  "$API/jobs" | py 'print(d["job"]["job_id"])')
echo "job $JOB queued; waiting for the worker ..."
until curl -sf "${AUTH[@]}" "$API/jobs/$JOB" | grep -qE '"status":"(completed|failed|cancelled)"'; do
  curl -sf "${AUTH[@]}" "$API/jobs/$JOB" | py 'p=d["job"]["progress"]; print("  ", p.get("stage"), p.get("analysed_frames"), "/", p.get("expected_samples"))'
  sleep 5
done
curl -sf "${AUTH[@]}" "$API/jobs/$JOB" | py '
j=d["job"]; a=j["coverage"].get("action_analysis") or {}; s=a.get("summary") or {}; m=j["metrics"]
print("status:", j["status"], (j["error"] or {}).get("code", ""))
print("events for review:", j["events_count"])
print("wall %.1f s, real-time factor %s, peak RSS %.0f MB" % (m.get("wall_time_ms",0)/1000, m.get("real_time_factor"), m.get("peak_rss_kb",0)/1024))
print("person visible:", s.get("person_visible_fraction"), " episodes:", s.get("episodes"))
for w in a.get("warnings", []): print("warning:", w["code"], "-", w["message"])
for c, r in (s.get("per_class") or {}).items(): print("  %-20s segments %3d  active %5.1f s" % (c, r["segments"], r["active_seconds"]))'
echo "open: http://127.0.0.1:8000/jobs/$JOB   then review at http://127.0.0.1:8000/review"
