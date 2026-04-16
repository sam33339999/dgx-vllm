#!/usr/bin/env python3
"""Fix doubled prefix bug in Qwen3Next/Qwen3.5 create_qkvz_proj method.

Bug: Both the caller and create_qkvz_proj append '.in_proj_qkvz' to prefix,
resulting in 'model.layers.X.linear_attn.in_proj_qkvz.in_proj_qkvz' which
doesn't match the quantization ignore list pattern.

Fix: Only change the prefix inside create_qkvz_proj method body, keeping
the caller's prefix=f"{prefix}.in_proj_qkvz" intact.

Compatible with vLLM v0.16.x (qwen3_next.py) and v0.19.x (qwen3_5.py).
"""

import os
import sys


def try_patch_file(path, model_name):
    """Attempt to patch a model file for doubled prefix bug."""
    if not os.path.exists(path):
        print(f"SKIP: {path} not found ({model_name} may not be in this vLLM version)")
        return False

    with open(path) as f:
        content = f.read()

    # Find create_qkvz_proj method and only fix the prefix inside it
    method_start = content.find("def create_qkvz_proj(")
    if method_start < 0:
        print(f"SKIP: create_qkvz_proj not found in {path}")
        return False

    # Find the next method/class definition to bound our search
    next_method = content.find("\ndef ", method_start + 1)
    if next_method < 0:
        next_method = len(content)

    method_body = content[method_start:next_method]

    # Check if the doubled prefix exists in the method body
    old_pattern = 'prefix=f"{prefix}.in_proj_qkvz"'
    if old_pattern in method_body:
        fixed_body = method_body.replace(old_pattern, "prefix=prefix", 1)
        content = content[:method_start] + fixed_body + content[next_method:]
        with open(path, "w") as f:
            f.write(content)
        print(f"Fix applied: {model_name} create_qkvz_proj doubled prefix removed")
        return True
    else:
        print(f"SKIP: pattern not found in {model_name} create_qkvz_proj (may already be fixed)")
        return False


# Try both Qwen3-Next (v0.16.x) and Qwen3.5 (v0.19.x) paths
vllm_root = "/app/vllm/vllm/model_executor/models"
candidates = [
    (f"{vllm_root}/qwen3_next.py", "Qwen3Next"),
    (f"{vllm_root}/qwen3_5.py", "Qwen3.5"),
]

patched = False
for path, name in candidates:
    if try_patch_file(path, name):
        patched = True

if not patched:
    print("No files needed patching (all already fixed or not present)")
    sys.exit(0)
