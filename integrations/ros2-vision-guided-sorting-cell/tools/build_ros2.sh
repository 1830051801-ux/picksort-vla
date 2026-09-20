#!/usr/bin/env bash
set -eo pipefail

SKIP_DEPS=false
if [[ "${1:-}" == "--skip-deps" ]]; then
  SKIP_DEPS=true
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--skip-deps]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

source /opt/ros/jazzy/setup.bash
if [[ "$SKIP_DEPS" != true ]]; then
  # rosdep's source cache is initialized by install_ros2_jazzy.sh.  Updating it
  # on every local build makes otherwise-offline rebuilds fail unnecessarily.
  rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y \
    --skip-keys ament_python
fi
colcon build --symlink-install --event-handlers console_direct+

echo
echo "Build complete."
echo "Run: source install/setup.bash"
echo "Then: ros2 launch sorting_cell_bringup sorting_cell.launch.py"
