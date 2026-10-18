#!/bin/bash
# Run quality checks: formatting (no changes) and tests
set -e
cd "$(dirname "$0")/.."

echo "==> Checking formatting (black)"
uv run black --check backend main.py

echo "==> Running tests (pytest)"
cd backend && uv run pytest
