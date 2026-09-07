#!/usr/bin/env bash
set -euo pipefail

FAST_LIO_WORKSPACE="${FAST_LIO_WORKSPACE:-$HOME/humble_ws}"
FAST_LIO_REVISION="${FAST_LIO_REVISION:-17b36d293a14df37d57e1751a337a32e2f164692}"

if [[ "$(. /etc/os-release && printf '%s' "$VERSION_CODENAME")" != "jammy" ]]; then
  echo "ROS 2 Humble binary setup requires Ubuntu 22.04 (jammy)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y software-properties-common curl
sudo add-apt-repository -y universe

if ! dpkg-query -W -f='${Status}' ros2-apt-source 2>/dev/null | grep -q 'install ok installed'; then
  ROS_APT_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
  ROS_APT_DEB="/tmp/ros2-apt-source_${ROS_APT_VERSION}.$$.deb"
  curl -fsSL -o "$ROS_APT_DEB" \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_VERSION}/ros2-apt-source_${ROS_APT_VERSION}.$(. /etc/os-release && printf '%s' "$UBUNTU_CODENAME")_all.deb"
  sudo dpkg -i "$ROS_APT_DEB"
  rm -f "$ROS_APT_DEB"
fi

sudo apt-get update
sudo apt-get install -y ros-humble-desktop ros-dev-tools ros-humble-rosbag2-storage-mcap

mkdir -p "$FAST_LIO_WORKSPACE/src"
if [[ ! -d "$FAST_LIO_WORKSPACE/src/spark-fast-lio/.git" ]]; then
  git clone https://github.com/MIT-SPARK/spark-fast-lio.git "$FAST_LIO_WORKSPACE/src/spark-fast-lio"
fi
git -C "$FAST_LIO_WORKSPACE/src/spark-fast-lio" fetch --tags origin
git -C "$FAST_LIO_WORKSPACE/src/spark-fast-lio" checkout --detach "$FAST_LIO_REVISION"

# ROS 2's generated setup scripts probe optional environment variables and are
# not compatible with bash nounset mode.
set +u
source /opt/ros/humble/setup.bash
set -u
if [[ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update
rosdep install --from-paths "$FAST_LIO_WORKSPACE/src" --ignore-src --rosdistro humble -r -y
cd "$FAST_LIO_WORKSPACE"
# Avoid linking the ROS component against an active Conda base environment.
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER \
  -u CONDA_PYTHON_EXE -u CONDA_EXE -u CMAKE_PREFIX_PATH -u LD_LIBRARY_PATH \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /bin/bash --noprofile --norc -c \
  'source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-up-to spark_fast_lio --cmake-clean-cache --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3'

set +u
source "$FAST_LIO_WORKSPACE/install/setup.bash"
set -u
ros2 pkg executables spark_fast_lio
echo "FAST-LIO Humble workspace is ready at $FAST_LIO_WORKSPACE"
