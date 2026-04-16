#!/bin/bash
# SM_121 FP8 Fix - Updated for vLLM v0.19.x architecture
# Patches: scaled_mm/cutlass.py to disable CUTLASS for SM_121
# Compatible with both v0.16.x and v0.19.x directory structures

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           SM_121 FP8 Backend Fix (v0.19.x)                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Search for the CUTLASS FP8 backend file in multiple possible locations
CUTLASS_CANDIDATES=(
    "/app/vllm/vllm/model_executor/layers/quantization/kernels/scaled_mm/cutlass.py"
    "/app/vllm/vllm/model_executor/layers/quantization/kernels/cutlass.py"
)

CUTLASS_FILE=""
for candidate in "${CUTLASS_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
        CUTLASS_FILE="$candidate"
        break
    fi
done

if [ -z "${CUTLASS_FILE}" ]; then
    echo "Searching for CUTLASS FP8 backend file..."
    CUTLASS_FILE=$(find /app/vllm -path "*/quantization/kernels/*cutlass*.py" -name "cutlass.py" 2>/dev/null | head -1)
fi

if [ -z "${CUTLASS_FILE}" ] || [ ! -f "${CUTLASS_FILE}" ]; then
    echo "ERROR: CUTLASS FP8 backend file not found!"
    echo "Searching for scaled_mm files..."
    find /app/vllm -name "cutlass.py" -path "*/quantization/*" 2>/dev/null | head -10
    exit 1
fi

echo "✓ Found: ${CUTLASS_FILE}"
echo ""

# Check if already patched
if grep -q "SM_121 (GB10)" "${CUTLASS_FILE}"; then
    echo "✅ Already patched, skipping"
    exit 0
fi

# Find the is_supported method in CutlassFP8ScaledMMLinearKernel class
echo "Patching CutlassFP8ScaledMMLinearKernel.is_supported()..."

# Find the line with "if not current_platform.is_cuda():"
PATCH_LINE=$(grep -n "if not current_platform.is_cuda():" "${CUTLASS_FILE}" | tail -1 | cut -d: -f1)

if [ -z "${PATCH_LINE}" ]; then
    echo "ERROR: Could not find 'if not current_platform.is_cuda():' in ${CUTLASS_FILE}"
    exit 1
fi

# Create the patch to insert BEFORE the is_cuda check
cat > /tmp/sm121_cutlass_patch.txt << 'EOFPATCH'
        # SM_121 (GB10) special case: CUTLASS FP8 has runtime incompatibility
        # Disable CUTLASS and fall back to PyTorch backend
        if compute_capability is not None and compute_capability == 121:
            return False, "SM_121 (GB10) not supported by CUTLASS - using PyTorch fallback"

        if not current_platform.is_cuda():
EOFPATCH

# Replace the line
awk -v line="${PATCH_LINE}" -v patch="$(cat /tmp/sm121_cutlass_patch.txt)" '
    NR == line {
        print patch
        next
    }
    { print }
' "${CUTLASS_FILE}" > "${CUTLASS_FILE}.new"

mv "${CUTLASS_FILE}.new" "${CUTLASS_FILE}"
echo "✓ Patched ${CUTLASS_FILE} at line ${PATCH_LINE}"

# Verify the patch
echo ""
echo "Verifying patch..."
if grep -q "SM_121 (GB10)" "${CUTLASS_FILE}"; then
    echo "✅ Patch applied successfully!"
    echo ""
    echo "Backend selection for SM_121:"
    echo "  - CUTLASS: Disabled (runtime incompatibility)"
    echo "  - PyTorch (torch._scaled_mm): Will be used as fallback"
else
    echo "❌ Patch verification failed!"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          SM_121 FP8 Fix Applied Successfully              ║"
echo "╚════════════════════════════════════════════════════════════╝"
