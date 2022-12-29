# Build Open3D from source
# http://www.open3d.org/docs/release/compilation.html

# Before running this:
conda activate my28brains

sudo apt-get install cmake

cd ~
git clone https://github.com/isl-org/Open3D

# install the dependencies and use CMake to build the project
cd ~/Open3D
source util/install_deps_ubuntu.sh

#
# echo which_python

mkdir build
cd build
cmake ..
make -j$(nproc)

# Install Python binding module

sudo make install

sudo make install-pip-package