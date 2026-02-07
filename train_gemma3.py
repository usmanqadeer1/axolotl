#!/usr/bin/env python3
"""
Gemma 3 Training Script with All Patches
"""
import sys
import os

print("\n" + "=" * 70)
print(" " * 20 + "GEMMA 3 TRAINING")
print("=" * 70)
print("\nApplying patches...\n")

# Apply all patches before importing axolotl
exec(open('/workspace/axolotl/patch_modelcard.py').read())
exec(open('/workspace/axolotl/gemma3_token_ids_patch.py').read())

print("\n" + "=" * 70)
print("All patches applied successfully! Starting training...")
print("=" * 70 + "\n")

# Now import and run Axolotl
from axolotl.cli.train import do_cli
import fire

if __name__ == "__main__":
    fire.Fire(do_cli)
