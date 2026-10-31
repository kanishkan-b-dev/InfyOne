@echo off
REM ===========================================================================
REM  InfyOne AI Chatbot - run.bat
REM  Loads all variables from .env into the current terminal session,
REM  then starts the FastAPI backend (which also serves the frontend).
REM
REM  Usage:
REM    run.bat              - start the server normally
REM    run.bat ingest       - run document ingestion into ChromaDB first, then start
REM    run.bat ingest-only  - only run ingestion, do not start the server
REM    run.bat test         - run the test suite
REM ===========================================================================

chcp 65001 >nul
setlocal EnableDelayedExpansion

REM -- Locate project root (same folder as this .bat file) -------------------
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM -- Check .env exists -----------------------------------------------------
if not exist ".env" (
    echo.
    echo  [ERROR] .env file not found in %ROOT%
    echo  Copy .env.example to .env and fill in your credentials:
    echo.
    echo      copy .env.example .env
    echo.
    exit /b 1
)

REM -- Load all KEY=VALUE pairs from .env into environment variables ----------
echo  Loading environment from .env ...
for /f "usebackq tokens=1,* delims== eol=#" %%A in (".env") do (
    set "line=%%A"
    REM Skip blank lines and comment lines
    if not "!line!"=="" (
        set "%%A=%%B"
    )
)
echo  [OK] Environment loaded.
echo.

REM -- Detect virtual environment --------------------------------------------
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    set "PIP=.venv\Scripts\pip.exe"
    set "UVICORN=.venv\Scripts\uvicorn.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
    set "PIP=venv\Scripts\pip.exe"
    set "UVICORN=venv\Scripts\uvicorn.exe"
) else (
    echo  [WARNING] No virtual environment found - .venv or venv.
    echo  Using system Python. It is recommended to create a venv:
    echo.
    echo      python -m venv .venv
    echo      .venv\Scripts\pip install -r requirements.txt
    echo.
    set "PYTHON=python"
    set "PIP=pip"
    set "UVICORN=uvicorn"
)

REM -- Parse command argument ------------------------------------------------
set "CMD=%~1"

REM ---------------------------------------------------------------------------
REM  COMMAND: test
REM ---------------------------------------------------------------------------
if /i "%CMD%"=="test" (
    echo  Running test suite ...
    echo  -------------------------------------------------
    %PYTHON% -m pytest tests/ -v
    exit /b %ERRORLEVEL%
)

REM ---------------------------------------------------------------------------
REM  COMMAND: ingest-only
REM ---------------------------------------------------------------------------
if /i "%CMD%"=="ingest-only" (
    echo  Running document ingestion ...
    echo  -------------------------------------------------
    %PYTHON% scripts/ingest.py
    echo.
    echo  [OK] Ingestion complete. Run run.bat to start the server.
    exit /b %ERRORLEVEL%
)

REM ---------------------------------------------------------------------------
REM  COMMAND: ingest (ingest first, then start server)
REM ---------------------------------------------------------------------------
if /i "%CMD%"=="ingest" (
    echo  Running document ingestion before starting server ...
    echo  -------------------------------------------------
    %PYTHON% scripts/ingest.py
    if %ERRORLEVEL% neq 0 (
        echo  [ERROR] Ingestion failed. Fix the errors above before starting.
        exit /b 1
    )
    echo.
)

REM ---------------------------------------------------------------------------
REM  DEFAULT - Start the FastAPI backend + frontend
REM ---------------------------------------------------------------------------

REM Read APP_HOST and APP_PORT from environment
if "%APP_HOST%"=="" set "APP_HOST=0.0.0.0"
if "%APP_PORT%"=="" set "APP_PORT=8000"

REM Determine reload flag
set "RELOAD_FLAG="
if /i "%APP_ENV%"=="development" set "RELOAD_FLAG=--reload"

echo  =======================================================
echo   App Name   : %APP_NAME%
echo   Server URL : http://localhost:%APP_PORT%
echo   API Docs   : http://localhost:%APP_PORT%/docs
echo   Mode       : %APP_ENV%
echo   Chroma Coll: %CHROMA_COLLECTION_NAME%
echo   Redis Host : %REDIS_HOST%
echo  =======================================================
echo.

%UVICORN% app.main:app --host %APP_HOST% --port %APP_PORT% %RELOAD_FLAG%

endlocal
