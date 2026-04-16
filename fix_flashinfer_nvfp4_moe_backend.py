#!/usr/bin/env python3
"""
Fix FlashInfer NVFP4 MoE backend auto-selection returning None for experts_cls.

Bug: In select_nvfp4_moe_backend(), the FlashInfer iteration path computes
k_cls = backend_to_kernel_cls(backend) but then returns (backend, None)
instead of (backend, k_cls). This causes an assertion failure in
CompressedTensorsW4A4Nvfp4MoEMethod.process_weights_after_loading()
when using VLLM_USE_FLASHINFER_MOE_FP4=1.

Fix: Return k_cls instead of None for non-TRTLLM FlashInfer backends.

Compatible with vLLM v0.16.x and v0.19.x.
"""

import sys
import os

def patch_nvfp4_oracle():
    # Find the file - try multiple paths
    vllm_root = "/app/vllm"
    candidates = [
        os.path.join(vllm_root, "vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"),
        os.path.join(vllm_root, "vllm/model_executor/layers/fused_moe/nvfp4.py"),
    ]

    target = None
    for candidate in candidates:
        if os.path.exists(candidate):
            target = candidate
            break

    if target is None:
        # Search for it
        import subprocess
        result = subprocess.run(
            ['find', vllm_root, '-name', 'nvfp4.py', '-path', '*/fused_moe/*'],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            target = result.stdout.strip().split('\n')[0]
        else:
            print("SKIP: NVFP4 MoE oracle file not found (may not exist in this vLLM version)")
            return

    with open(target, "r") as f:
        content = f.read()

    # Check if already patched
    if "return backend, k_cls" in content:
        print("Already patched")
        return

    old = """                if supported:
                    logger.info_once(_make_log_backend(backend), scope="local")
                    return backend, None
                else:
                    logger.debug_once(
                        _make_log_unsupported(backend, reason), scope="local"
                    )"""

    new = """                if supported:
                    logger.info_once(_make_log_backend(backend), scope="local")
                    return backend, k_cls
                else:
                    logger.debug_once(
                        _make_log_unsupported(backend, reason), scope="local"
                    )"""

    if old not in content:
        print(f"SKIP: Target pattern not found in {target}")
        print("This bug may have been fixed upstream in this vLLM version")
        return

    content = content.replace(old, new)

    with open(target, "w") as f:
        f.write(content)

    print(f"Patched {target}: FlashInfer NVFP4 MoE backend now returns k_cls")

if __name__ == "__main__":
    patch_nvfp4_oracle()
