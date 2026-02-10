#!/bin/bash
# Stop immediately if any command fails
set -e

# Enable Conda environment
source /home/kirito/miniconda/bin/activate automl_gpu_env

echo "---------------------------------------------------------"
echo "Fixing Compiler Incompatibility & Linker Segfaults"
echo "---------------------------------------------------------"

# 1. Install GCC 12 & Binutils & Newer Sysroot
# The Segfault often happens due to incompatible binutils/ld. 
# We upgrade sysroot to 2.28 and ensure binutils is matched.
echo "Updating build toolchain..."
conda install -y -c conda-forge \
    gcc_linux-64=12 \
    gxx_linux-64=12 \
    binutils_linux-64 \
    sysroot_linux-64=2.28 \
    cuda-cudart

# 2. Set Compiler Paths
export CC=/home/kirito/miniconda/envs/automl_gpu_env/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/kirito/miniconda/envs/automl_gpu_env/bin/x86_64-conda-linux-gnu-g++
export AR=/home/kirito/miniconda/envs/automl_gpu_env/bin/x86_64-conda-linux-gnu-ar
export LD=/home/kirito/miniconda/envs/automl_gpu_env/bin/x86_64-conda-linux-gnu-ld

echo "Using Compiler Version:"
$CC --version | head -n 1

# 3. LOCATE CUDA SHARED LIBRARY & USE DYNAMIC LINKING
# We are moving away from the static hack because it caused a segfault.
echo "Locating CUDA runtime libraries..."
CUDART_SHARED=$(find $CONDA_PREFIX -name libcudart.so | head -n 1)

if [ -n "$CUDART_SHARED" ]; then
    CUDART_DIR=$(dirname "$CUDART_SHARED")
    echo "Found SHARED CUDA Libs at: $CUDART_DIR"
    
    # Export env vars for dynamic linker
    export LIBRARY_PATH=$CUDART_DIR:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDART_DIR:$LD_LIBRARY_PATH
    export LDFLAGS="-L$CUDART_DIR $LDFLAGS"
    
    # Remove the fake static lib if we created it (it might be causing the crash)
    if [ -L "$CUDART_DIR/libcudart_static.a" ]; then
        echo "Removing fake static symlink to avoid confusion..."
        rm "$CUDART_DIR/libcudart_static.a"
    fi
    
    # Force CMake to use this specific shared library
    BUILD_FLAGS="--config-settings=cmake.define.USE_CUDA=ON"
    BUILD_FLAGS="$BUILD_FLAGS --config-settings=cmake.define.CUDA_CUDART_LIBRARY=$CUDART_SHARED"
    BUILD_FLAGS="$BUILD_FLAGS --config-settings=cmake.define.CUDA_USE_STATIC_CUDA_RUNTIME=OFF"
else
    echo "CRITICAL ERROR: No CUDA runtime found."
    exit 1
fi

# 4. Clean previous failed builds
set +e
pip uninstall -y lightgbm
set -e

# 5. Install LightGBM
echo "---------------------------------------------------------"
echo "Starting LightGBM Compilation (Clean Dynamic Mode)..."
echo "---------------------------------------------------------"

pip install \
  --force-reinstall \
  --no-binary lightgbm \
  --no-cache \
  $BUILD_FLAGS \
  lightgbm

echo "---------------------------------------------------------"
echo "DONE. Please restart the application."
echo "---------------------------------------------------------"
