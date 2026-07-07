@echo off
setlocal
set PROJECT=C:\Users\1\Desktop\institution-alpha
set OUTPUT_DIR=C:\Users\1\Desktop\2026_level2_ops
set TEMP_DIR=%OUTPUT_DIR%\temp_extract
set LOG_DIR=%OUTPUT_DIR%\logs
set LOG_FILE=%LOG_DIR%\pipeline_full.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] Starting DBSCAN pipeline for 2026 > "%LOG_FILE%"
echo   Plan: %PROJECT%\data\processed\stock_universe\process_plan.csv >> "%LOG_FILE%"
echo   Output: %OUTPUT_DIR% >> "%LOG_FILE%"

cd /d "%PROJECT%"
python scripts\run_level2_process_plan.py --plan data/processed/stock_universe/process_plan.csv --start-year 2026 --end-year 2026 --limit 0 --temp-dir "%TEMP_DIR%" --output-dir "%OUTPUT_DIR%" 2>&1 >> "%LOG_FILE%"

echo [%date% %time%] Pipeline finished (exit=%ERRORLEVEL%) >> "%LOG_FILE%"
