#!/usr/bin/env bash
set -e

echo "=== Step 1/4: pinned av ==="
pip install "av==16.1.0" --quiet --no-deps

echo "=== Step 2/4: accelerate ==="
pip install "accelerate>=0.26.0" --quiet

echo "=== Step 3/4: lerobot[pi] from git ==="
pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git" \
    --quiet \
    --extra-index-url https://pypi.org/simple

echo "=== Step 4/4: scikit-learn ==="
pip install "scikit-learn>=1.3.0" --quiet

echo ""
echo "All installs done. NOW restart the runtime before continuing."
echo "  Runtime > Restart runtime  (or Ctrl+M .) then run check_versions.py."
