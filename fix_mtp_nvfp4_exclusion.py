#!/usr/bin/env python3
"""
Fix MTP layer exclusion for ModelOpt NVFP4 quantization.

Bug: The NVFP4 checkpoint exclude list has 'mtp.layers.0*' to skip MTP decoder
layers, but ALL MTP weights (including mtp.fc, mtp.layers.0.self_attn.*, etc.)
are stored as BF16 in the checkpoint. The pattern 'mtp.layers.0*' does NOT
match 'mtp.fc', so the fc layer gets initialized as FP4 quantized while the
checkpoint has BF16 weights — causing a shape mismatch assertion failure.

Fix: In is_layer_excluded(), if any exclude pattern starts with 'mtp.' and the
current prefix also starts with 'mtp.', exclude the entire MTP module. All MTP
weights in NVFP4 checkpoints are BF16 (unquantized).

Compatible with vLLM v0.16.x and v0.19.x.
"""

import sys
import os


def patch_modelopt():
    # Try multiple possible paths
    candidates = [
        "/app/vllm/vllm/model_executor/layers/quantization/modelopt.py",
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
            ['find', '/app/vllm', '-name', 'modelopt.py', '-path', '*/quantization/*'],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            target = result.stdout.strip().split('\n')[0]
        else:
            print("SKIP: modelopt.py not found (may not be needed in this vLLM version)")
            return

    with open(target, "r") as f:
        content = f.read()

    # Check if already patched
    if 'prefix.startswith("mtp.")' in content:
        print("Already patched")
        return

    # The bug is in is_layer_excluded(). The wildcard 'mtp.layers.0*' only
    # matches mtp.layers.0.XXX but misses mtp.fc and other MTP layers.
    # Since ALL MTP weights are BF16, we exclude the entire mtp.* prefix.
    old = """        # modelopt exclude modules are not simple strings, they are wildcards
        for wildcard_pattern in self.exclude_modules:
            if fnmatch(prefix, wildcard_pattern):
                return True

        return False"""

    new = """        # modelopt exclude modules are not simple strings, they are wildcards
        for wildcard_pattern in self.exclude_modules:
            if fnmatch(prefix, wildcard_pattern):
                return True

        # All MTP weights in NVFP4 checkpoints are BF16 (unquantized).
        # The exclude list only has 'mtp.layers.0*' which misses 'mtp.fc'.
        # If any exclude pattern targets mtp.*, exclude all mtp.* layers.
        if prefix.startswith("mtp."):
            for wildcard_pattern in self.exclude_modules:
                if wildcard_pattern.startswith("mtp."):
                    return True

        return False"""

    if old not in content:
        # Try to find the function with slight variations
        if "def is_layer_excluded" in content and "exclude_modules" in content:
            print(f"WARNING: is_layer_excluded found in {target} but pattern differs")
            print("The function signature may have changed in this vLLM version")
            print("Attempting regex-based patch...")

            import re
            # Try to insert the mtp check before "return False" in is_layer_excluded
            pattern = r'(def is_layer_excluded.*?)(        return False)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                insert_point = match.start(2)
                mtp_check = """        # All MTP weights in NVFP4 checkpoints are BF16 (unquantized).
        if prefix.startswith("mtp."):
            for wildcard_pattern in self.exclude_modules:
                if wildcard_pattern.startswith("mtp."):
                    return True

"""
                content = content[:insert_point] + mtp_check + content[insert_point:]
                with open(target, "w") as f:
                    f.write(content)
                print(f"Patched {target}: ALL MTP layers now excluded from NVFP4 (regex method)")
                return
            else:
                print("ERROR: Could not find insertion point")
                sys.exit(1)
        else:
            print(f"SKIP: is_layer_excluded or exclude_modules not found in {target}")
            print("This function may have been refactored in this vLLM version")
            return

    content = content.replace(old, new)

    with open(target, "w") as f:
        f.write(content)

    print(f"Patched {target}: ALL MTP layers now excluded from NVFP4")


if __name__ == "__main__":
    patch_modelopt()
