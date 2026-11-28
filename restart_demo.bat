@echo off
rem EdgePark - replay the camera video from the beginning with a fresh edge node.
rem The dashboard removes the old violations automatically when the node restarts.
cd /d "%~dp0"
echo Restarting the camera stream and the edge node ...
docker compose restart camera edge-node
echo Done. Violations appear again after about 60-90 seconds.
ping -n 5 127.0.0.1 >nul
