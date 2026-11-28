@echo off
rem EdgePark - stop all demo containers.
cd /d "%~dp0"
echo Stopping camera, edge node, MQTT broker and dashboard ...
docker compose down
echo Done.
ping -n 4 127.0.0.1 >nul
