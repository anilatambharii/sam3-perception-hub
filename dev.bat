@echo off
setlocal enabledelayedexpansion

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="download-models" goto download-models
if "%1"=="dev" goto dev
if "%1"=="dev-api" goto dev-api
if "%1"=="dev-ui" goto dev-ui
if "%1"=="test" goto test
if "%1"=="test-unit" goto test-unit
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="docker-build" goto docker-build
if "%1"=="docker-up" goto docker-up
if "%1"=="docker-down" goto docker-down
if "%1"=="demo-privacy" goto demo-privacy
if "%1"=="demo-warehouse" goto demo-warehouse
if "%1"=="demo-ar" goto demo-ar
if "%1"=="demo-content" goto demo-content
if "%1"=="clean" goto clean
goto help

:help
echo SAM3 Perception Hub - Development Commands
echo.
echo Setup:
echo   dev.bat install          Install all dependencies
echo   dev.bat download-models  Download model checkpoints
echo.
echo Development:
echo   dev.bat dev              Start development servers (all)
echo   dev.bat dev-api          Start API servers only
echo   dev.bat dev-ui           Start UI only
echo   dev.bat test             Run all tests
echo   dev.bat test-unit        Run unit tests only
echo   dev.bat lint             Run linters
echo   dev.bat format           Format code
echo.
echo Docker:
echo   dev.bat docker-build     Build all Docker images
echo   dev.bat docker-up        Start all services
echo   dev.bat docker-down      Stop all services
echo.
echo Demos:
echo   dev.bat demo-privacy     Run privacy-preserving analytics demo
echo   dev.bat demo-warehouse   Run warehouse analytics demo
echo   dev.bat demo-ar          Run AR reconstruction demo
echo   dev.bat demo-content     Run content production demo
echo.
echo Other:
echo   dev.bat clean            Clean build artifacts
goto end

:install
echo Installing dependencies...
pip install -e ".[dev]"
cd ui\playground && npm install
cd ..\..
pre-commit install
goto end

:download-models
python scripts\download_models.py
goto end

:dev
echo Starting all development servers...
echo Note: Use Ctrl+C to stop all servers
start "Perception API" cmd /k "cd services\perception-api && uvicorn sam3_perception.main:app --reload --port 8080"
start "Reconstruction API" cmd /k "cd services\reconstruction-api && uvicorn sam3_reconstruction.main:app --reload --port 8081"
start "Agent Bridge" cmd /k "cd services\agent-bridge && uvicorn sam3_agent.main:app --reload --port 8082"
start "UI Playground" cmd /k "cd ui\playground && npm run dev"
echo All servers started in separate windows
goto end

:dev-api
echo Starting API servers...
start "Perception API" cmd /k "cd services\perception-api && uvicorn sam3_perception.main:app --reload --port 8080"
start "Reconstruction API" cmd /k "cd services\reconstruction-api && uvicorn sam3_reconstruction.main:app --reload --port 8081"
start "Agent Bridge" cmd /k "cd services\agent-bridge && uvicorn sam3_agent.main:app --reload --port 8082"
echo API servers started in separate windows
goto end

:dev-ui
cd ui\playground && npm run dev
goto end

:test
pytest tests\ -v --cov=services --cov-report=term-missing
goto end

:test-unit
pytest tests\unit\ -v
goto end

:lint
ruff check services\ tests\
mypy services\
goto end

:format
ruff format services\ tests\
ruff check --fix services\ tests\
goto end

:docker-build
docker compose build
goto end

:docker-up
docker compose up -d
goto end

:docker-down
docker compose down
goto end

:demo-privacy
echo Running Privacy-Preserving Analytics Demo...
python examples\scripts\run_demo_privacy.py
goto end

:demo-warehouse
echo Running Warehouse Analytics Demo...
python examples\scripts\run_demo_warehouse.py
goto end

:demo-ar
echo Running AR Reconstruction Demo...
python examples\scripts\run_demo_ar.py
goto end

:demo-content
echo Running Content Production Demo...
python examples\scripts\run_demo_content.py
goto end

:clean
echo Cleaning build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .mypy_cache rmdir /s /q .mypy_cache
if exist .ruff_cache rmdir /s /q .ruff_cache
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage del .coverage
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
echo Cleanup complete
goto end

:end
endlocal
