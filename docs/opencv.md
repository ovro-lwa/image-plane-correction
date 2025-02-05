# Installation

```bash
# Ensure that CUDA is already installed on the system.

# Clone the main OpenCV repository
git clone git@github.com:opencv/opencv.git

# Clone the extra OpenCV modules (since these contain the CUDA-accelerated optical flow algorithms of interest)
git@github.com:opencv/opencv_contrib.git

# switch both repositories to the same branch
cd opencv
git checkout 4.x
cd ..

cd opencv_contrib
git checkout 4.x
cd ..

# Build OpenCV with CUDA support and python bindings
cd opencv
mkdir build
cd build

# If using conda, enter the environment you wish to install into (and make sure numpy is already installed)
conda activate <ENVIRONMENT>

cmake -D OPENCV_EXTRA_MODULES_PATH=<PATH_TO_OPENCV_CONTRIB>/modules \
  -D CMAKE_INSTALL_PREFIX=<WHERE_TO_INSTALL> \
  -D WITH_CUDA=ON \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D WITH_OPENGL=ON \
  -D WITH_CUBLAS=1 \
  -D BUILD_opencv_python3=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D INSTALL_C_EXAMPLES=OFF \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D BUILD_EXAMPLES=ON \
  -D PYTHON_VERSION=311 \
  -D PYTHON3_EXECUTABLE=$(which python3) \
  -D PYTHON_INCLUDE_DIR=$(python3 -c "from sysconfig import get_path; print(get_path('include'))") \
  -D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
  -D PYTHON_LIBRARY=$(python3 -c "from sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
  -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
  -D PYTHON3_PACKAGES_PATH=$(python3 -c "from sysconfig import get_path; print(get_path('platlib'))") \
  ..

# Make sure the output includes something like:
# --  Python 3:
# --     Interpreter:                 <PYTHON_PATH>/bin/python3 (ver 3.11)
# --     Libraries:                   <PYTHON_PATH>/lib/libpython3.11.a (ver 3.11.0)
# --     Limited API:                 NO
# --     numpy:                       <PYTHON_PATH>/lib/python3.11/site-packages/numpy/core/include (ver 1.26.4)
# --     install path:                <PYTHON_PATH>/lib/python3.11/site-packages/cv2/python-3.11

make -j5

# may require sudo, depending on if you're installing system-wide or locally
make install

# Test if the installation worked and has CUDA support
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"  # should print 1 or greater
```
