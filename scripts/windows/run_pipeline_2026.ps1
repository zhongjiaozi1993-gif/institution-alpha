# DBSCAN Pipeline for 2026 — Full Run
# Execute on Windows: powershell -File run_pipeline_2026.ps1

$ErrorActionPreference = "Continue"

$PROJECT = "C:\Users\1\Desktop\institution-alpha"
$OUTPUT_DIR = "C:\Users\1\Desktop\2026_level2_ops"
$TEMP_DIR = "$OUTPUT_DIR\temp_extract"
$LOG_DIR = "$OUTPUT_DIR\logs"
$FAILED_FILE = "$OUTPUT_DIR\failed_dates.csv"

New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

$PLAN_CSV = "$PROJECT\data\processed\stock_universe\process_plan.csv"
$PROCESS_SCRIPT = "$PROJECT\scripts\run_level2_process_plan.py"
$LOG_FILE = "$LOG_DIR\pipeline_full.log"

$START_TIME = Get-Date
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting DBSCAN pipeline for 2026" | Tee-Object -Append -FilePath $LOG_FILE
"  Plan: $PLAN_CSV" | Tee-Object -Append -FilePath $LOG_FILE
"  Output: $OUTPUT_DIR" | Tee-Object -Append -FilePath $LOG_FILE
"  Temp: $TEMP_DIR" | Tee-Object -Append -FilePath $LOG_FILE

python $PROCESS_SCRIPT `
    --plan $PLAN_CSV `
    --start-year 2026 `
    --end-year 2026 `
    --limit 0 `
    --temp-dir $TEMP_DIR `
    --output-dir $OUTPUT_DIR `
    2>&1 | Tee-Object -Append -FilePath $LOG_FILE

$EXIT_CODE = $LASTEXITCODE
$ELAPSED = [math]::Round(((Get-Date) - $START_TIME).TotalMinutes, 1)

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Pipeline finished (exit=$EXIT_CODE, elapsed=${ELAPSED}min)" | Tee-Object -Append -FilePath $LOG_FILE

# Save summary
$outputFiles = Get-ChildItem -Path "$OUTPUT_DIR\2026" -Filter "level2_ops_*.csv" -ErrorAction SilentlyContinue
$count = if ($outputFiles) { $outputFiles.Count } else { 0 }
"  Output files: $count" | Tee-Object -Append -FilePath $LOG_FILE

if ($EXIT_CODE -ne 0) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] PIPELINE FAILED (exit=$EXIT_CODE)" | Tee-Object -Append -FilePath $LOG_FILE
}
