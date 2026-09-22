#!/usr/bin/env bash
# Read-only check of Modal apps and running containers in the active environment.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run modal app list --json
uv run modal container list --json
