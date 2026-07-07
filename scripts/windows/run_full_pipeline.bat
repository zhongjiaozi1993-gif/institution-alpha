@echo off
cd /d C:\Users\1\Desktop\institution-alpha
set LOG=C:\Users\1\Desktop\2026_level2_ops\logs\pipeline_full.log
echo [%date% %time%] START DBSCAN pipeline (80 days) >> "%LOG%"
python scripts\run_level2_process_plan.py --plan data/processed/stock_universe/process_plan.csv --start-year 2026 --end-year 2026 --limit 0 --temp-dir C:\Users\1\Desktop\2026_level2_ops\temp_extract --output-dir C:\Users\1\Desktop\2026_level2_ops >> "%LOG%" 2>&1
echo [%date% %time%] DONE (exit=%ERRORLEVEL%) >> "%LOG%"
