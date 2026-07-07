# Extract only index universe (HS300+ZZ500+ZZ1000) stocks from .7z archives
# Process: list archive -> filter to universe -> extract per-stock -> delete .7z
#
# Usage: powershell -File extract_universe.ps1 [-Year 2025|2026] [-DryRun]

param(
    [string]$Year = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$SEVEN_ZIP = "C:\Users\1\Desktop\7zr.exe"
$UNIVERSE_FILE = "C:\Users\1\Desktop\institution-alpha\data\processed\stock_universe\index_universe.txt"
$ARCHIVE_ROOT = "C:\Users\1\Desktop"
$OUTPUT_ROOT = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$LOG_FILE = "C:\Users\1\Desktop\extract_universe.log"

# Load universe
$UNIVERSE = @{}
Get-Content $UNIVERSE_FILE | Where-Object { $_ -match '^\d{6}$' } | ForEach-Object { $UNIVERSE[$_] = $true }
Write-Host "Universe: $($UNIVERSE.Count) stocks"

# Find all .7z archives
$years = if ($Year) { @($Year) } else { @("2025", "2026") }
$archives = @()
foreach ($y in $years) {
    $yearDir = Join-Path $ARCHIVE_ROOT $y
    if (Test-Path $yearDir) {
        $found = Get-ChildItem -Path $yearDir -Recurse -Filter "*.7z" -ErrorAction SilentlyContinue
        $archives += $found
        Write-Host "  $y : $($found.Count) archives"
    }
}

$total = $archives.Count
Write-Host "Total archives: $total"
Write-Host "Output root: $OUTPUT_ROOT"
if ($DryRun) { Write-Host "DRY RUN MODE" -ForegroundColor Yellow }
Write-Host ""

$processed = 0
$skipped = 0
$failed = 0
$freedMB = 0
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($archive in $archives) {
    $processed++
    $archivePath = $archive.FullName
    $archiveSizeMB = [math]::Round($archive.Length / 1MB, 0)

    # Extract date from archive name
    $dateMatch = [regex]::Match($archive.Name, '(\d{8})')
    if (-not $dateMatch.Success) {
        Write-Host "[$processed/$total] SKIP $($archive.Name): no date" -ForegroundColor Gray
        $skipped++
        continue
    }
    $dateStr = $dateMatch.Groups[1].Value

    $elapsed = [math]::Round($stopwatch.Elapsed.TotalMinutes, 1)
    Write-Host "[$processed/$total] $dateStr ($archiveSizeMB MB, ${elapsed}min)" -ForegroundColor White

    # List stocks in this archive that are in our universe
    if ($DryRun) {
        Write-Host "  [DRY-RUN] would extract universe stocks from $($archive.Name)" -ForegroundColor Yellow
        continue
    }

    $listOutput = & $SEVEN_ZIP l "$archivePath" -ba 2>&1
    $stocksInArchive = @{}
    foreach ($line in $listOutput) {
        if ($line -match '(\d{6})\.(SZ|SH)') {
            $code = $Matches[1]
            if ($UNIVERSE.ContainsKey($code)) {
                $suffix = $Matches[2]
                $stocksInArchive[$code] = $suffix
            }
        }
    }

    $nStocks = $stocksInArchive.Count
    if ($nStocks -eq 0) {
        Write-Host "  SKIP: no universe stocks in this archive" -ForegroundColor Gray
        $skipped++
        continue
    }

    Write-Host "  Found $nStocks universe stocks, extracting..."

    $extractOK = 0
    $extractFail = 0

    foreach ($code in $stocksInArchive.Keys) {
        $suffix = $stocksInArchive[$code]
        $archivePattern = "$dateStr\$code.$suffix"
        $outputDir = Join-Path $OUTPUT_ROOT "$code\raw\$dateStr"
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

        $result = & $SEVEN_ZIP x "$archivePath" "$archivePattern\*" -o"$outputDir" -aoa -y 2>&1

        # Check if files were extracted
        $extractedFiles = Get-ChildItem $outputDir -File -ErrorAction SilentlyContinue
        if ($extractedFiles.Count -gt 0) {
            $extractOK++
        } else {
            $extractFail++
            Write-Host "    FAIL: $code.$suffix" -ForegroundColor Red
        }
    }

    Write-Host "  Extracted: $extractOK OK / $extractFail FAIL"

    # Delete archive after successful extraction
    if ($extractFail -eq 0) {
        Remove-Item $archivePath -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path $archivePath)) {
            $freedMB += $archiveSizeMB
            Write-Host "  Deleted archive (+$archiveSizeMB MB)" -ForegroundColor Green
        } else {
            Write-Host "  WARN: could not delete archive" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Archive kept (had failures)" -ForegroundColor Yellow
        $failed++
    }
}

$stopwatch.Stop()
$totalMin = [math]::Round($stopwatch.Elapsed.TotalMinutes, 1)
$freedGB = [math]::Round($freedMB / 1024, 1)

Write-Host ""
Write-Host "========================================"
Write-Host "DONE in ${totalMin}min"
Write-Host "  Processed: $processed"
Write-Host "  Skipped:   $skipped"
Write-Host "  Failed:    $failed"
Write-Host "  Freed:     $freedGB GB"
Write-Host "========================================"
