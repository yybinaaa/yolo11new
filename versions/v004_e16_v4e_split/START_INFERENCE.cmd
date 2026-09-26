@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_INFERENCE.ps1" %*
pause
