#!/bin/bash
# ============================================================================
# End-to-end Training Pipeline with enzyme.autodiff
# ============================================================================
#
# Training goal:
#   y = X @ W + b
#   X = [[1,0],[0,1],[1,1],[1,-1]]
#   y_true = [[2],[3],[5],[-1]]
# optimal solution: 
#   W = [[2],[3]], b = [0]
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENZYME_BUILD="/home/zuo/code/Enzyme-fork/enzyme/build/Enzyme/MLIR"
LLVM_BUILD="/home/zuo/code/llvm-project/build/bin"

EOPT="${ENZYME_BUILD}/enzymemlir-opt"
MOPT="${LLVM_BUILD}/mlir-opt"
MCR="${LLVM_BUILD}/mlir-cpu-runner"
TRAINING_MLIR="${SCRIPT_DIR}/training_autodiff.mlir"
RUNTIME_SO="${SCRIPT_DIR}/execution/runtime.so"
RUNTIME_C="${SCRIPT_DIR}/execution/runtime.c"
LOWERING_DIR="${SCRIPT_DIR}/lowering"

mkdir -p "${LOWERING_DIR}"

echo "=== Step 0: Build runtime library ==="
if [ ! -f "${RUNTIME_SO}" ] || [ "${RUNTIME_C}" -nt "${RUNTIME_SO}" ]; then
    "${LLVM_BUILD}/clang" -shared -fPIC -o "${RUNTIME_SO}" "${RUNTIME_C}"
    echo "  Built: ${RUNTIME_SO}"
else
    echo "  Runtime library up-to-date."
fi

echo ""
echo "=== Step 1: --enzyme pass (lowering enzyme.autodiff op) ==="
echo "  input: training_autodiff.mlir (including enzyme.autodiff @forward_loss)"
"${EOPT}" \
    --enzyme \
    --remove-unnecessary-enzyme-ops \
    --enzyme-simplify-math \
    --canonicalize \
    "${TRAINING_MLIR}" > "${LOWERING_DIR}/after_enzyme_pass.mlir"
echo "  output: ${LOWERING_DIR}/after_enzyme_pass.mlir"
echo "  enzyme.autodiff → func.call @diffeforward_loss"

echo ""
echo "=== Step 2: --convert-enzyme-to-memref (降级 enzyme cache ops) ==="
"${EOPT}" \
    --convert-enzyme-to-memref \
    "${LOWERING_DIR}/after_enzyme_pass.mlir" > "${LOWERING_DIR}/after_enzyme_to_memref.mlir"
echo "  output: ${LOWERING_DIR}/after_enzyme_to_memref.mlir"

echo ""
echo "=== Step 3: MLIR lowering passes ==="
"${MOPT}" --allow-unregistered-dialect \
    --convert-scf-to-cf \
    --expand-strided-metadata \
    --lower-affine \
    --finalize-memref-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --convert-index-to-llvm \
    --reconcile-unrealized-casts \
    "${LOWERING_DIR}/after_enzyme_to_memref.mlir" > "${LOWERING_DIR}/llvm_dialect_autodiff.mlir"
echo "  output: ${LOWERING_DIR}/llvm_dialect_autodiff.mlir"

echo ""
echo "=== Step 4: JIT training ==="
echo "  train: y = X @ W + b"
echo "  X = [[1,0],[0,1],[1,1],[1,-1]]"
echo "  y_true = [[2],[3],[5],[-1]]"
echo "  optimal: W = [[2],[3]], b = [0]"
echo "  learning rate: 0.05, Epochs: 500"
echo ""
echo "  Loss (every 50 epochs):"
"${MCR}" \
    --shared-libs="${RUNTIME_SO}" \
    --entry-point-result=void \
    "${LOWERING_DIR}/llvm_dialect_autodiff.mlir"

echo ""
echo "=== Done ==="
echo " last 3 values: W[0,0], W[1,0], b[0] (目标: 2.0, 3.0, 0.0)"
