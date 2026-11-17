#!/usr/bin/env bash
set -euo pipefail

# Run in a Python 3.11 environment with a compatible CUDA toolchain.
pip3 install vllm==0.6.3
pip3 install ray
pip install -e .
pip install -r requirements.txt
pip3 install flash-attn --no-build-isolation
pip install wandb IPython matplotlib
# Install MLEBench and prepare competition data separately; see README.md.
