#!/bin/bash
# Format all Python code with black
set -e
cd "$(dirname "$0")/.."
uv run black backend main.py
