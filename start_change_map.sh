#!/bin/bash
# Double-click to start the app (macOS / Linux).
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Get it from https://www.python.org/downloads/"
  read -r -p "Press Enter to close."
  exit 1
fi
python3 launch.py || read -r -p "Something went wrong (see above). Press Enter to close."
