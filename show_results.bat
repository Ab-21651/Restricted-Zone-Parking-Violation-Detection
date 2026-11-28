@echo off
rem EdgePark - re-run the alert evaluation (base rule vs. variants A-E, thresholds
rem 10/30/60 s, dev and test videos) from the cached tracks and print the table.
cd /d "%~dp0"
title EdgePark results
if not exist ".venv\Scripts\python.exe" (
    echo Python environment .venv not found. See README "Setup".
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\eval_alerts.py
echo.
echo Full results: docs\RESULTS.md    Figures: outputs\figures
start "" "docs\RESULTS.md"
pause
