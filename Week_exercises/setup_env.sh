#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

. ".venv/bin/activate"
python -m pip --isolated install --index-url https://pypi.org/simple --upgrade pip
python -m pip --isolated install --index-url https://pypi.org/simple -r requirements.txt
python -m ipykernel install --user --name in3050-lessons --display-name "IN3050 Lessons"

echo
echo "Environment is ready."
echo "Run: .venv/bin/jupyter lab"
