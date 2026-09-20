#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run and verify the complete headless ROS 2 / Gazebo sorting-cell demo.

Usage: bash tools/run_ros2_demo.sh [options]

Options:
  --timeout SECONDS   Live verifier timeout (default: 240).
  --output-dir PATH   Evidence directory (default: results/ros2-runtime-<UTC>).
  --skip-build        Reuse the existing install/ workspace.
  -h, --help          Show this help.

The command always uses real color perception, disables Gazebo GUI and RViz,
and stops every process it started before returning.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT_SECONDS="${SORTING_CELL_TIMEOUT:-240}"
OUTPUT_DIR=""
BUILD_WORKSPACE=true

while (($#)); do
  case "$1" in
    --timeout)
      [[ $# -ge 2 ]] || { echo "--timeout requires a value" >&2; exit 2; }
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --skip-build)
      BUILD_WORKSPACE=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--timeout must be a positive integer" >&2
  exit 2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="$PROJECT_ROOT/results/ros2-runtime-$run_stamp"
elif [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
if [[ "$OUTPUT_DIR" == "/" ]]; then
  echo "Refusing to use the filesystem root as the evidence directory" >&2
  exit 2
fi

VERIFICATION_JSON="$OUTPUT_DIR/verification.json"
LAUNCH_LOG="$OUTPUT_DIR/launch.log"
VERIFIER_LOG="$OUTPUT_DIR/verifier.log"
METRICS_CSV="$OUTPUT_DIR/metrics.csv"
rm -f -- "$VERIFICATION_JSON" "$LAUNCH_LOG" "$VERIFIER_LOG" "$METRICS_CSV"
touch "$LAUNCH_LOG" "$VERIFIER_LOG"

ROS_DISTRO_NAME="${SORTING_CELL_ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/$ROS_DISTRO_NAME/setup.bash"
if [[ ! -r "$ROS_SETUP" ]]; then
  echo "ROS setup was not found at $ROS_SETUP" >&2
  echo "Install ROS 2 first with tools/install_ros2_jazzy.sh." >&2
  exit 2
fi

# shellcheck disable=SC1090
set +u
source "$ROS_SETUP"
set -u
for command_name in colcon python3 ros2 setsid timeout; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    exit 2
  fi
done

cd "$PROJECT_ROOT"
if [[ "$BUILD_WORKSPACE" == true ]]; then
  echo "Building ROS 2 workspace..."
  colcon build --symlink-install --event-handlers console_direct+
fi
if [[ ! -r "$PROJECT_ROOT/install/setup.bash" ]]; then
  echo "Workspace setup is missing: $PROJECT_ROOT/install/setup.bash" >&2
  echo "Run without --skip-build, or run tools/build_ros2.sh first." >&2
  exit 2
fi
# shellcheck disable=SC1091
set +u
source "$PROJECT_ROOT/install/setup.bash"
set -u

if ! ros2 pkg prefix sorting_cell_bringup >/dev/null 2>&1; then
  echo "sorting_cell_bringup is not available in the sourced workspace" >&2
  exit 2
fi

# Keep this run isolated from unrelated ROS graphs while allowing callers to
# select a specific domain when several CI jobs run on the same host.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RCUTILS_COLORIZED_OUTPUT="${RCUTILS_COLORIZED_OUTPUT:-0}"
# Gazebo Transport is independent of DDS. Give each wrapper invocation its own
# partition so an already-running simulator cannot contaminate this evidence.
export GZ_PARTITION="${GZ_PARTITION:-sorting_cell_${UID}_$$}"

LAUNCH_PID=""
VERIFIER_PID=""

group_is_alive() {
  local leader_pid="$1"
  [[ -n "$leader_pid" ]] && kill -0 -- "-$leader_pid" 2>/dev/null
}

stop_group() {
  local leader_pid="$1"
  local label="$2"
  local signal_name
  local attempts

  [[ -n "$leader_pid" ]] || return 0
  if ! group_is_alive "$leader_pid"; then
    wait "$leader_pid" 2>/dev/null || true
    return 0
  fi

  # Let ros2 launch orchestrate an orderly child shutdown first. Signalling the
  # whole group with SIGINT makes every rclpy process tear DDS down at once.
  kill -INT -- "$leader_pid" 2>/dev/null || true
  attempts=75
  while ((attempts > 0)) && group_is_alive "$leader_pid"; do
    sleep 0.2
    attempts=$((attempts - 1))
  done

  for signal_name in TERM KILL; do
    group_is_alive "$leader_pid" || break
    kill "-$signal_name" -- "-$leader_pid" 2>/dev/null || true
    case "$signal_name" in
      TERM) attempts=25 ;;
      KILL) attempts=5 ;;
    esac
    while ((attempts > 0)) && group_is_alive "$leader_pid"; do
      sleep 0.2
      attempts=$((attempts - 1))
    done
    group_is_alive "$leader_pid" || break
  done
  if group_is_alive "$leader_pid"; then
    echo "Warning: $label process group $leader_pid did not stop cleanly" >&2
  fi
  wait "$leader_pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_group "$VERIFIER_PID" "verifier"
  stop_group "$LAUNCH_PID" "ROS launch"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_failure_evidence() {
  local reason="$1"
  [[ ! -s "$VERIFICATION_JSON" ]] || return 0
  WRAPPER_ERROR="$reason" TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
    python3 - "$VERIFICATION_JSON" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "passed": False,
    "wrapper_error": os.environ["WRAPPER_ERROR"],
    "timeout_s": int(os.environ["TIMEOUT_SECONDS"]),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

mark_artifact_failure() {
  local reason="$1"
  WRAPPER_ERROR="$reason" python3 - "$VERIFICATION_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    payload = {}
payload["passed"] = False
payload["wrapper_error"] = os.environ["WRAPPER_ERROR"]
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

LAUNCH_COMMAND=(
  ros2 launch sorting_cell_bringup sorting_cell.launch.py
  rviz:=false
  gazebo_gui:=false
  perception:=color
  "metrics_csv:=$METRICS_CSV"
)
if [[ -z "${DISPLAY:-}" ]]; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "No DISPLAY is available and xvfb-run is not installed." >&2
    echo "Install the xvfb package for camera rendering on headless Linux." >&2
    exit 2
  fi
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  LAUNCH_COMMAND=(
    xvfb-run -a -s "-screen 0 1280x720x24"
    "${LAUNCH_COMMAND[@]}"
  )
fi

echo "Starting headless Gazebo with real color perception..."
setsid "${LAUNCH_COMMAND[@]}" >"$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!

# Start immediately: the verifier subscribes before the first sort cycle while
# its controller-service wait naturally covers Gazebo and controller startup.
EXTERNAL_TIMEOUT_SECONDS=$((TIMEOUT_SECONDS + 20))
setsid timeout --signal=INT --kill-after=10s "${EXTERNAL_TIMEOUT_SECONDS}s" \
  python3 "$PROJECT_ROOT/tools/verify_ros2_demo.py" \
    --timeout "$TIMEOUT_SECONDS" \
    --output "$VERIFICATION_JSON" \
  >"$VERIFIER_LOG" 2>&1 &
VERIFIER_PID=$!

FINISHED_PID=""
set +e
wait -n -p FINISHED_PID "$LAUNCH_PID" "$VERIFIER_PID"
FIRST_STATUS=$?
set -e

RUN_STATUS=0
if [[ "${FINISHED_PID:-}" == "$LAUNCH_PID" ]]; then
  RUN_STATUS=1
  write_failure_evidence "ROS launch exited before verification completed (exit $FIRST_STATUS)"
  echo "ROS launch exited before verification completed." >&2
  tail -n 80 "$LAUNCH_LOG" >&2 || true
else
  RUN_STATUS=$FIRST_STATUS
  if [[ $RUN_STATUS -ne 0 ]]; then
    if [[ $RUN_STATUS -eq 124 ]]; then
      write_failure_evidence "Live verification exceeded the bounded timeout"
    else
      write_failure_evidence "Live verifier exited with status $RUN_STATUS"
    fi
  fi
fi

if [[ -s "$VERIFIER_LOG" ]]; then
  cat "$VERIFIER_LOG"
fi

if [[ $RUN_STATUS -eq 0 && ! -s "$METRICS_CSV" ]]; then
  RUN_STATUS=1
  mark_artifact_failure "Verification passed but metrics.csv was not produced"
  echo "Verification did not produce a non-empty metrics CSV." >&2
fi

echo
echo "ROS 2 runtime evidence:"
echo "  JSON:    $VERIFICATION_JSON"
echo "  metrics: $METRICS_CSV"
echo "  launch:  $LAUNCH_LOG"

if [[ $RUN_STATUS -eq 0 ]]; then
  echo "Headless ROS 2 / Gazebo verification passed."
else
  echo "Headless ROS 2 / Gazebo verification failed (exit $RUN_STATUS)." >&2
fi
exit "$RUN_STATUS"
