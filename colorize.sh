#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
  printf 'Usage: %s INPUT_FOLDER [--output OUTPUT_FOLDER] [--recursive]\n' "$0" >&2
  exit 2
fi
if [[ ! -x "$script_dir/.venv/bin/python" ]]; then
  "${PYTHON:-python3}" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else "Use Python 3.11 or 3.12, for example PYTHON=python3.11 ./colorize.sh input")'
  "${PYTHON:-python3}" -m venv "$script_dir/.venv"
fi
if ! "$script_dir/.venv/bin/python" - "$script_dir/requirements.txt" <<'PY'
import importlib.metadata, sys
try:
    for line in open(sys.argv[1]):
        package, version = line.strip().split('==')
        if importlib.metadata.version(package) != version:
            raise ValueError(package)
except (importlib.metadata.PackageNotFoundError, ValueError):
    sys.exit(1)
PY
then
  "$script_dir/.venv/bin/python" -m ensurepip --upgrade
  "$script_dir/.venv/bin/python" -m pip install --no-cache-dir -r "$script_dir/requirements.txt"
fi
exec "$script_dir/.venv/bin/python" "$script_dir/colorize.py" "$@"
