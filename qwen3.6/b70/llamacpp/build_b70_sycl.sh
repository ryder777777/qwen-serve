#!/usr/bin/env bash
# Build llama.cpp with SYCL backend for Intel Arc B70 (Battlemage / Xe2).
#
# Prerequisites:
#   - Intel oneAPI 2026 (DPC++/SYCL compiler, oneMKL)
#   - CMake >= 3.22
#   - git
#
# Install oneAPI 2026 on the host:
#   wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo apt-key add -
#   echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
#   sudo apt update && sudo apt install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel
#   source /opt/intel/oneapi/setvars.sh
#
# Or use the Intel oneAPI Docker image:
#   docker run -it --device /dev/dri:/dev/dri --group-add 987 \
#     -v /path/to/models:/models -v $(pwd):/work -w /work \
#     intel/oneapi-basekit:2026.0.0-devel bash

set -euo pipefail

source /opt/intel/oneapi/setvars.sh 2>/dev/null || true

REPO_DIR="${REPO_DIR:-llama.cpp}"
BUILD_DIR="${BUILD_DIR:-build-sycl}"

if [ ! -d "${REPO_DIR}" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Battlemage (B70/B60) is Xe2. The SYCL backend targets the Intel GPU
# automatically via oneAPI. GGML_SYCL_DEVICE=0 selects the first XPU.
cmake -B "${BUILD_DIR}" \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_TARGET=INTEL \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --config Release -j "$(nproc)"

echo ""
echo "Build complete. Binary: ${REPO_DIR}/${BUILD_DIR}/bin/llama-server"
echo ""
echo "To convert the model to GGUF:"
echo "  python3 ${REPO_DIR}/convert_hf_to_gguf.py /models/Qwen3.6-35B-A3B --outtype q4_k_m --outfile /models/Qwen3.6-35B-A3B-Q4_K_M.gguf"
echo ""
echo "To launch:"
echo "  GGML_SYCL_DEVICE=0 ${REPO_DIR}/${BUILD_DIR}/bin/llama-server -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 99 -fa 1 -c 8192 --port 8040"
