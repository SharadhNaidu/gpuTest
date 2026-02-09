@echo off
REM AI Desktop Assessment - One Command Runner for Windows
REM Usage: run.bat or double-click this file
python "%~dp0run.py" %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo Press any key to exit...
    pause >nul
)
