@echo off
rem Double-click to start the app (Windows).
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 launch.py
) else (
  python launch.py
)
if errorlevel 1 (
  echo.
  echo Something went wrong - see the messages above. If Python is missing, get it from https://www.python.org/downloads/
  pause
)
