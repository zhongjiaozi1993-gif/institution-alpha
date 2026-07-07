<#
.SYNOPSIS
  DBSCAN Pipeline for 2026 Raw L2 Data (PowerShell)
.DESCRIPTION
  Processes C:\Users\1\Desktop\2026\{month}\*.7z files through
  the DBSCAN clustering pipeline, outputting daily level2_ops CSV files
  to C:\Users\1\Desktop\2026_level2_ops\

  Features:
  - Per-month processing
  - Per-day output files
  - Skip if output already exists (resume-safe)
  - Logging with timestamps
  - Failed files recorded separately
  - Memory-safe: processes one day at a time

.PARAMETER DryRun
  If specified, only print commands without executing

.PARAMETER Month
  Specific month to process (e.g. "202601"). Default: all 5 months
#>

param(
    [switch]$DryRun,
    [string]$Month = ""
)

$ErrorActionPreference = "Continue"
$PROJECT = "C:\Users\1\Desktop\institution-alpha"
$ARCHIVE_ROOT = "C:\Users\1\Desktop\2026"
$OUTPUT_DIR = "C:\Users\1\Desktop\2026_level2_ops"
$TEMP_DIR = "C:\Users\1\Desktop\2026_level2_ops\temp_extract"
$LOG_DIR = "C:\Users\1\Desktop\2026_level2_ops\logs"
$FAILED_FILE = "C:\Users\1\Desktop\2026_level2_ops\failed_dates.csv"
$PYTHON = "python"

# Ensure output directories
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

$MONTHS = if ($Month) { @($Month) } else { @("202601", "202602", "202603", "202604", "202605") }

$TOTAL_START = Get-Date

# ============================================================
# Step 1: Generate process_plan for 2026
# ============================================================
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "STEP 1: Generate process_plan for 2026" -ForegroundColor Cyan
Write-Host "================================================================"

$PLAN_SCRIPT = Join-Path $PROJECT "scripts\plan_level2_universe.py"
$PLAN_LOG = Join-Path $LOG_DIR "plan_generation.log"

$planArgs = @(
    $PLAN_SCRIPT,
    "--archive-root", $ARCHIVE_ROOT,
    "--years", "2026",
    "--max-stocks", "300",
    "--cache-scan", (Join-Path $OUTPUT_DIR "archive_scan_cache.json")
)

Write-Host "Command: $PYTHON $($planArgs -join ' ')"
if (-not $DryRun) {
    & $PYTHON $planArgs 2>&1 | Tee-Object -FilePath $PLAN_LOG
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: plan generation failed! Check $PLAN_LOG" -ForegroundColor Red
        exit 1
    }
    Write-Host "Plan generated successfully." -ForegroundColor Green
} else {
    Write-Host "[DRY RUN] Would execute plan generation" -ForegroundColor Yellow
}

# ============================================================
# Step 2: Process each month
# ============================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "STEP 2: Run DBSCAN pipeline per month" -ForegroundColor Cyan
Write-Host "================================================================"

$PROCESS_SCRIPT = Join-Path $PROJECT "scripts\run_level2_process_plan.py"
$PLAN_CSV = Join-Path $PROJECT "data\processed\stock_universe\process_plan.csv"

$ALL_FAILED = @()
$TOTAL_DAYS = 0
$COMPLETED_DAYS = 0
$SKIPPED_DAYS = 0

foreach ($m in $MONTHS) {
    Write-Host ""
    Write-Host "--- Processing month: $m ---" -ForegroundColor Yellow
    $MONTH_LOG = Join-Path $LOG_DIR "process_$m.log"
    $MONTH_START = Get-Date

    # Count days in this month's archive dir
    $monthDir = Join-Path $ARCHIVE_ROOT $m
    if (-not (Test-Path $monthDir)) {
        Write-Host "  WARNING: Directory not found: $monthDir" -ForegroundColor Red
        continue
    }
    $archiveFiles = Get-ChildItem -Path $monthDir -Filter "*.7z"
    Write-Host "  Found $($archiveFiles.Count) archive files" -ForegroundColor Gray

    foreach ($archive in $archiveFiles) {
        $dateMatch = [regex]::Match($archive.Name, '(\d{8})')
        if (-not $dateMatch.Success) { continue }
        $dateStr = $dateMatch.Groups[1].Value
        $outputFile = Join-Path $OUTPUT_DIR "level2_ops_$dateStr.csv"
        $TOTAL_DAYS++

        # Skip if output already exists
        if (Test-Path $outputFile) {
            Write-Host "  [$dateStr] SKIP: output already exists" -ForegroundColor Gray
            $SKIPPED_DAYS++
            continue
        }

        Write-Host "  [$dateStr] Processing..." -ForegroundColor White
        $DAY_START = Get-Date

        $processArgs = @(
            $PROCESS_SCRIPT,
            "--plan", $PLAN_CSV,
            "--start-year", "2026",
            "--end-year", "2026",
            "--limit", "1",
            "--temp-dir", $TEMP_DIR,
            "--output-dir", $OUTPUT_DIR
        )

        if ($DryRun) {
            Write-Host "    [DRY RUN] $PYTHON $($processArgs -join ' ')" -ForegroundColor Yellow
            continue
        }

        try {
            $result = & $PYTHON $processArgs 2>&1
            $exitCode = $LASTEXITCODE

            $result | Out-File -Append -FilePath $MONTH_LOG

            if ($exitCode -eq 0 -and (Test-Path $outputFile)) {
                $fileSize = (Get-Item $outputFile).Length
                $dayElapsed = [math]::Round(((Get-Date) - $DAY_START).TotalSeconds, 1)
                Write-Host "    OK: $dateStr ($fileSize bytes, ${dayElapsed}s)" -ForegroundColor Green
                $COMPLETED_DAYS++
            } else {
                Write-Host "    FAILED: $dateStr (exit code: $exitCode)" -ForegroundColor Red
                $ALL_FAILED += [PSCustomObject]@{
                    date = $dateStr
                    archive = $archive.FullName
                    exit_code = $exitCode
                    error = ($result -join "`n")
                }
            }
        } catch {
            Write-Host "    EXCEPTION: $dateStr - $_" -ForegroundColor Red
            $ALL_FAILED += [PSCustomObject]@{
                date = $dateStr
                archive = $archive.FullName
                exit_code = -1
                error = $_.Exception.Message
            }
        }

        # Clean temp after each day
        if (Test-Path $TEMP_DIR) {
            Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
        }
    }

    $monthElapsed = [math]::Round(((Get-Date) - $MONTH_START).TotalMinutes, 1)
    Write-Host "  Month $m done in ${monthElapsed}min" -ForegroundColor Cyan
}

# ============================================================
# Step 3: Summary
# ============================================================
$TOTAL_ELAPSED = [math]::Round(((Get-Date) - $TOTAL_START).TotalMinutes, 1)

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================"
Write-Host "  Total days:    $TOTAL_DAYS"
Write-Host "  Completed:     $COMPLETED_DAYS"
Write-Host "  Skipped:       $SKIPPED_DAYS"
Write-Host "  Failed:        $($ALL_FAILED.Count)"
Write-Host "  Total time:    ${TOTAL_ELAPSED}min"
Write-Host "  Output dir:    $OUTPUT_DIR"

# Save failed dates
if ($ALL_FAILED.Count -gt 0) {
    $ALL_FAILED | Export-Csv -Path $FAILED_FILE -NoTypeInformation -Encoding UTF8
    Write-Host "  Failed list:   $FAILED_FILE" -ForegroundColor Red
}

# Show output stats
$outputFiles = Get-ChildItem -Path $OUTPUT_DIR -Filter "level2_ops_*.csv" -ErrorAction SilentlyContinue
if ($outputFiles) {
    $totalSize = [math]::Round(($outputFiles | Measure-Object -Property Length -Sum).Sum / 1MB, 2)
    Write-Host "  Output files:  $($outputFiles.Count) ($totalSize MB)"
}

Write-Host ""
if ($ALL_FAILED.Count -eq 0) {
    Write-Host "ALL DONE - No failures!" -ForegroundColor Green
} else {
    Write-Host "DONE with $($ALL_FAILED.Count) failures - check $FAILED_FILE" -ForegroundColor Yellow
}
