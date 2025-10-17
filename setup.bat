@echo off
REM RAG Workshop Setup Script for Windows
REM This script automates the environment setup using UV

echo ==================================
echo RAG Workshop Environment Setup
echo ==================================
echo.

REM Check if uv is installed
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo UV is not installed.
    echo Installing UV...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo UV installed successfully
) else (
    echo UV is already installed
)

echo.
echo Installing dependencies from pyproject.toml...
echo    (This will create a virtual environment and install all packages)
uv sync

echo.
echo Setting up environment variables...
if not exist .env (
    copy .env.example .env
    echo Created .env file from template
    echo Please edit .env and add your Azure OpenAI credentials
) else (
    echo .env file already exists
)

echo.
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo Next steps:
echo 1. Activate the virtual environment:
echo    .venv\Scripts\activate
echo.
echo 2. Edit .env with your credentials:
echo    notepad .env
echo.
echo 3. Start Jupyter:
echo    jupyter notebook
echo.
echo 4. Navigate to RAG_hf_v4\ to access the demos
echo.
pause
