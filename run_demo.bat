@echo off
rem ===================================================================
rem  EdgePark live demo - double-click to start everything.
rem  Starts Docker Desktop if needed, then the camera, edge node,
rem  MQTT broker and dashboard, and opens the dashboard in the browser.
rem ===================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
title EdgePark live demo

if "%CAM%"=="" set CAM=ISLab-07

echo.
echo  EdgePark - restricted-zone parking violation detection (live demo)
echo  -------------------------------------------------------------------
echo  Tip: laptop on the CHARGER and power mode "Best performance".
echo.

rem ---- 1. make sure Docker is running --------------------------------
docker info >nul 2>&1
if not errorlevel 1 goto docker_ok

echo [1/3] Starting Docker Desktop ...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
set /a tries=0
:wait_docker
ping -n 6 127.0.0.1 >nul
docker info >nul 2>&1
if not errorlevel 1 goto docker_ok
set /a tries+=1
if !tries! equ 18 (
    echo       Docker is not up after 90 s - repairing it ^(stale socket files^) ...
    powershell -NoProfile -ExecutionPolicy Bypass -File "tools\fix-docker.ps1"
)
if !tries! lss 42 goto wait_docker
echo.
echo  ERROR: Docker did not start. Open Docker Desktop, wait for "Engine running",
echo         then double-click run_demo.bat again.
pause
exit /b 1

:docker_ok
echo [1/3] Docker is running.

rem ---- 2. start the four containers ------------------------------------
echo [2/3] Starting camera, edge node, MQTT broker and dashboard (video %CAM%) ...
docker compose up -d
if errorlevel 1 (
    echo.
    echo  ERROR: docker compose failed - see the message above.
    pause
    exit /b 1
)

rem ---- 3. open the dashboard -------------------------------------------
echo [3/3] Opening the dashboard ...
set /a tries=0
:wait_dash
ping -n 3 127.0.0.1 >nul
curl -s -o nul http://localhost:8000 && goto dash_ok
set /a tries+=1
if !tries! lss 15 goto wait_dash
:dash_ok
start "" http://localhost:8000

echo.
echo  ===================================================================
echo   Running.  Dashboard: http://localhost:8000
echo.
echo   What happens (video %CAM%):
echo     ~60-70 s   VIOLATION  motorcycle parked in the fire lane
echo     ~70-95 s   VIOLATION  SUV that drives in and parks
echo     ~130 s     video loops: the SUV "leaves" -^> CLEARED event
echo.
echo   Other buttons:  restart_demo.bat  = replay the video from the start
echo                   show_results.bat  = print the evaluation results
echo                   stop_demo.bat     = stop everything
echo  ===================================================================
echo.
echo  Below is the live log of the edge node (close this window any time;
echo  the system keeps running until you run stop_demo.bat).
echo.
docker compose logs -f --tail 5 edge-node 2>nul | findstr /v /c:"h264" /c:"mmco"
