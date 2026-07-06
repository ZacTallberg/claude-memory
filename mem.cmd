@echo off
REM `mem` launcher (cmd.exe / shell): forwards all args to `python -m claudemem`.
REM Uses the project venv so dependencies (fastembed, fastapi, psycopg) resolve.
REM PYTHONUTF8=1 forces UTF-8 I/O so transcript/emoji content never crashes on cp1252.
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m claudemem %*
exit /b %ERRORLEVEL%
