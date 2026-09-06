@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv" (
    python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip --isolated install --index-url https://pypi.org/simple --upgrade pip
python -m pip --isolated install --index-url https://pypi.org/simple -r requirements.txt
python -m ipykernel install --user --name in3050-lessons --display-name "IN3050 Lessons"

echo.
echo Environment is ready.
echo Run: .venv\Scripts\jupyter lab
