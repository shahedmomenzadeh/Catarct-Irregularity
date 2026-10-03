@echo off
setlocal enabledelayedexpansion
title Cataract Surgery Irregularity Benchmark Runner (CSI-Bench)

cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo ===========================================================================
echo   CATARACT SURGERY IRREGULARITY BENCHMARK RUNNER (CSI-Bench)
echo   3-Class Intraoperative Classification (Normal, Lens Irreg, Pupil Cont)
echo   100%% Deterministic Scoring - One-by-One Evaluation Protocol
echo ===========================================================================
echo.

:: 1. Check uv package manager
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] uv not found in PATH. Checking local app data...
    if exist "%LOCALAPPDATA%\programs\uv\uv.exe" (
        set "PATH=%LOCALAPPDATA%\programs\uv;%PATH%"
    ) else if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    ) else (
        echo [INFO] Installing uv package manager...
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        set "PATH=%LOCALAPPDATA%\programs\uv;%PATH%"
    )
)

where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] uv could not be installed. Please install manually from https://astral.sh/uv
    pause
    exit /b 1
)

:: 2. Parse Arguments or defaults
set "MODEL_FAMILY=api"
if not "%~1"=="" set "MODEL_FAMILY=%~1"

if "%MODEL_FAMILY%"=="--dry-run" goto RUN_DRY
if "%MODEL_FAMILY%"=="dry-run" goto RUN_DRY

set "OUTPUT_DIR=%SCRIPT_DIR%\results"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
set "DATASET_ROOT=%SCRIPT_DIR%\Irregularity_dataset"

echo Model Family Selection : %MODEL_FAMILY%
echo Output Directory       : %OUTPUT_DIR%
echo Dataset Root           : %DATASET_ROOT%
echo.

goto DISPATCH

:RUN_DRY
echo [INFO] Executing dry-run dataset verification...
python "%SCRIPT_DIR%\main.py" --dry-run --dataset-root "%DATASET_ROOT%"
pause
exit /b 0

:DISPATCH
if "%MODEL_FAMILY%"=="all" goto RUN_API
if "%MODEL_FAMILY%"=="openai_api" goto RUN_API
if "%MODEL_FAMILY%"=="gemini" goto RUN_API
if "%MODEL_FAMILY%"=="api" goto RUN_API
if "%MODEL_FAMILY%"=="qwen3vl" goto RUN_QWEN
if "%MODEL_FAMILY%"=="hulumed" goto RUN_HULUMED
if "%MODEL_FAMILY%"=="lingshu" goto RUN_LINGSHU
if "%MODEL_FAMILY%"=="mage_vl" goto RUN_MAGEVL

:RUN_API
echo ---------------------------------------------------------------------------
echo [1/5] API Inference: ag/gemini-3.8-flash on 9router
echo ---------------------------------------------------------------------------
if not exist "%SCRIPT_DIR%\.venv-api" (
    echo Creating .venv-api environment...
    uv venv "%SCRIPT_DIR%\.venv-api" --python 3.11
)
set "API_PY=%SCRIPT_DIR%\.venv-api\Scripts\python.exe"
uv pip install --python "%API_PY%" -r "%SCRIPT_DIR%\requirements\requirements-api.txt"
"%API_PY%" "%SCRIPT_DIR%\main.py" --model-family api --model-id "ag/gemini-3.8-flash" --api-base-url "http://localhost:20128/v1" --api-timeout 600 --api-retries 5 --output-dir "%OUTPUT_DIR%"
if "%MODEL_FAMILY%"=="api" goto FINISHED
if "%MODEL_FAMILY%"=="openai_api" goto FINISHED
if "%MODEL_FAMILY%"=="gemini" goto FINISHED

:RUN_HULUMED
echo ---------------------------------------------------------------------------
echo [2/5] HuluMed Family: Hulu-Med-7B and Hulu-Med-4B
echo ---------------------------------------------------------------------------
if not exist "%SCRIPT_DIR%\.venv-hulumed" (
    echo Creating .venv-hulumed environment...
    uv venv "%SCRIPT_DIR%\.venv-hulumed" --python 3.12
)
set "HULU_PY=%SCRIPT_DIR%\.venv-hulumed\Scripts\python.exe"
uv pip install --python "%HULU_PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "%HULU_PY%" -r "%SCRIPT_DIR%\requirements\requirements-hulumed.txt"
"%HULU_PY%" "%SCRIPT_DIR%\main.py" --model-family hulumed --model-id "ZJU-AI4H/Hulu-Med-7B" --frame-size 224 --output-dir "%OUTPUT_DIR%"
"%HULU_PY%" "%SCRIPT_DIR%\main.py" --model-family hulumed --model-id "ZJU-AI4H/Hulu-Med-4B" --frame-size 224 --output-dir "%OUTPUT_DIR%"
if "%MODEL_FAMILY%"=="hulumed" goto FINISHED

:RUN_QWEN
echo ---------------------------------------------------------------------------
echo [3/5] Qwen3-VL Family: 2B, 4B, 8B Instruct
echo ---------------------------------------------------------------------------
if not exist "%SCRIPT_DIR%\.venv-qwen3vl" (
    echo Creating .venv-qwen3vl environment...
    uv venv "%SCRIPT_DIR%\.venv-qwen3vl" --python 3.12
)
set "QWEN_PY=%SCRIPT_DIR%\.venv-qwen3vl\Scripts\python.exe"
uv pip install --python "%QWEN_PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "%QWEN_PY%" -r "%SCRIPT_DIR%\requirements\requirements-qwen3vl.txt"
"%QWEN_PY%" "%SCRIPT_DIR%\main.py" --model-family qwen3vl --model-id "Qwen/Qwen3-VL-2B-Instruct" --output-dir "%OUTPUT_DIR%"
"%QWEN_PY%" "%SCRIPT_DIR%\main.py" --model-family qwen3vl --model-id "Qwen/Qwen3-VL-4B-Instruct" --output-dir "%OUTPUT_DIR%"
"%QWEN_PY%" "%SCRIPT_DIR%\main.py" --model-family qwen3vl --model-id "Qwen/Qwen3-VL-8B-Instruct" --output-dir "%OUTPUT_DIR%"
if "%MODEL_FAMILY%"=="qwen3vl" goto FINISHED

:RUN_LINGSHU
echo ---------------------------------------------------------------------------
echo [4/5] Lingshu-7B MLLM
echo ---------------------------------------------------------------------------
if not exist "%SCRIPT_DIR%\.venv-qwen3vl" (
    uv venv "%SCRIPT_DIR%\.venv-qwen3vl" --python 3.12
)
set "QWEN_PY=%SCRIPT_DIR%\.venv-qwen3vl\Scripts\python.exe"
uv pip install --python "%QWEN_PY%" -r "%SCRIPT_DIR%\requirements\requirements-qwen3vl.txt"
"%QWEN_PY%" "%SCRIPT_DIR%\main.py" --model-family lingshu --model-id "lingshu-medical-mllm/Lingshu-7B" --output-dir "%OUTPUT_DIR%"
if "%MODEL_FAMILY%"=="lingshu" goto FINISHED

:RUN_MAGEVL
echo ---------------------------------------------------------------------------
echo [5/5] Mage-VL
echo ---------------------------------------------------------------------------
if not exist "%SCRIPT_DIR%\.venv-magevl" (
    echo Creating .venv-magevl environment...
    uv venv "%SCRIPT_DIR%\.venv-magevl" --python 3.12
)
set "MAGE_PY=%SCRIPT_DIR%\.venv-magevl\Scripts\python.exe"
uv pip install --python "%MAGE_PY%" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "%MAGE_PY%" -r "%SCRIPT_DIR%\requirements\requirements-magevl.txt"
"%MAGE_PY%" "%SCRIPT_DIR%\main.py" --model-family mage_vl --model-id "microsoft/Mage-VL" --output-dir "%OUTPUT_DIR%"
if "%MODEL_FAMILY%"=="mage_vl" goto FINISHED

:FINISHED
echo.
echo ===========================================================================
echo  EVALUATION FINISHED!
echo  Results saved to: %OUTPUT_DIR%
echo ===========================================================================
pause
