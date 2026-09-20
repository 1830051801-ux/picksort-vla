#!/usr/bin/env bash
set -euo pipefail

if ! grep -q "24.04" /etc/os-release; then
  echo "This installer expects Ubuntu 24.04." >&2
  exit 1
fi

sudo rm -f /etc/apt/sources.list.d/ros2.list /usr/share/keyrings/ros-archive-keyring.gpg
sudo apt update
sudo apt install -y locales software-properties-common curl gnupg
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe

ROS_APT_SOURCE_VERSION="1.2.0"
ROS_APT_SOURCE_SHA256="0804d9b13db770eb87019be414cd78378835228ad5fa801fc88758596dd8f7e5"
ROS_APT_SOURCE_DEB="$(mktemp --suffix=.deb)"
curl -fsSL \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb" \
  -o "$ROS_APT_SOURCE_DEB"
ACTUAL_SHA256="$(sha256sum "$ROS_APT_SOURCE_DEB" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$ROS_APT_SOURCE_SHA256" ]]; then
  echo "ros2-apt-source checksum mismatch: $ACTUAL_SHA256" >&2
  rm -f "$ROS_APT_SOURCE_DEB"
  exit 1
fi
sudo apt install -y "$ROS_APT_SOURCE_DEB"
rm -f "$ROS_APT_SOURCE_DEB"

sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-trajectory-controller \
  ros-jazzy-ros2controlcli \
  ros-jazzy-xacro \
  python3-colcon-common-extensions \
  python3-opencv \
  python3-rosdep \
  python3-vcstool \
  xvfb

sudo rosdep init 2>/dev/null || true
rosdep update

if ! grep -q "/opt/ros/jazzy/setup.bash" "$HOME/.bashrc"; then
  echo "source /opt/ros/jazzy/setup.bash" >> "$HOME/.bashrc"
fi

echo
echo "ROS 2 Jazzy, Gazebo Harmonic and ros2_control are installed."
echo "Open a new terminal, then run tools/build_ros2.sh from the project."
