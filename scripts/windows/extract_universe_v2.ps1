# Extract only index universe stocks from .7z archives (v2 — batch extraction)
# Uses @filelist for single 7zr call per archive
#
# Usage: powershell -File extract_universe_v2.ps1 [-Year 2025|2026] [-Month 01-12] [-DryRun] [-LogFile path]

param(
    [string]$Year = "",
    [string]$Month = "",
    [switch]$DryRun,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Continue"
$SEVEN_ZIP = "C:\Users\1\Desktop\7zr.exe"
$UNIVERSE_FILE = "C:\Users\1\Desktop\institution-alpha\data\processed\stock_universe\index_universe.txt"
$ARCHIVE_ROOT = "C:\Users\1\Desktop"
$OUTPUT_ROOT = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$TEMP_LIST = "C:\Users\1\Desktop\temp_extract_list.txt"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $msg"
    Write-Host $line
    if ($LogFile) { Add-Content -Path $LogFile -Value $line -Encoding UTF8 }
}

Log "=== Extract Universe v2 ==="

# Load universe — code -> suffix mapping (0xxxxx/3xxxxx -> SZ, 6xxxxx -> SH)
$UNIVERSE = @{}
Get-Content $UNIVERSE_FILE | Where-Object { $_ -match '^\d{6}$' } | ForEach-Object {
    $code = $_
    $suffix = if ($code.StartsWith("6")) { "SH" } else { "SZ" }
    $UNIVERSE[$code] = $suffix
}
Log "Universe: $($UNIVERSE.Count) stocks (SZ: $($UNIVERSE.Values.Where({$_ -eq 'SZ'}).Count), SH: $($UNIVERSE.Values.Where({$_ -eq 'SH'}).Count))"

# Find all .7z archives
$years = if ($Year) { @($Year) } else { @("2025", "2026") }
$archives = @()
foreach ($y in $years) {
    $yearDir = Join-Path $ARCHIVE_ROOT $y
    if (-not (Test-Path $yearDir)) { continue }
    if ($Month) {
        $monthDir = Join-Path $yearDir "$y$Month"
        if (Test-Path $monthDir) {
            $found = Get-ChildItem -Path $monthDir -Filter "*.7z" -ErrorAction SilentlyContinue
            $archives += $found
            Log "  $y-$Month : $($found.Count) archives"
        }
    } else {
        $found = Get-ChildItem -Path $yearDir -Recurse -Filter "*.7z" -ErrorAction SilentlyContinue
        $archives += $found
        Log "  $y : $($found.Count) archives"
    }
}

$total = $archives.Count
Log "Total archives: $total"
if ($DryRun) { Log "DRY RUN MODE" }
Log ""

$processed = 0
$deleted = 0
$failed = 0
$freedGB = 0
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($archive in $archives) {
    $processed++
    $archivePath = $archive.FullName
    $archiveSizeGB = [math]::Round($archive.Length / 1GB, 2)

    $dateMatch = [regex]::Match($archive.Name, '(\d{8})')
    if (-not $dateMatch.Success) {
        Log "[$processed/$total] SKIP $($archive.Name): no date"
        continue
    }
    $dateStr = $dateMatch.Groups[1].Value

    $elapsed = [math]::Round($stopwatch.Elapsed.TotalMinutes, 1)
    $diskFree = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    Log "[$processed/$total] $dateStr ($archiveSizeGB GB, disk free: ${diskFree}GB, elapsed: ${elapsed}min)"

    if ($DryRun) { continue }

    # Step 1: List contents, find universe stocks
    $listOutput = & $SEVEN_ZIP l "$archivePath" -ba 2>&1
    $stocksInArchive = @{}
    foreach ($line in $listOutput) {
        if ($line -match '(\d{6})\.(SZ|SH)') {
            $code = $Matches[1]
            if ($UNIVERSE.ContainsKey($code)) {
                $stocksInArchive[$code] = $Matches[2]
            }
        }
    }

    $nStocks = $stocksInArchive.Count
    if ($nStocks -eq 0) {
        Log "  -> SKIP (no universe stocks)"
        continue
    }

    # Step 2: Build file list for batch extraction
    $fileList = @()
    foreach ($code in $stocksInArchive.Keys) {
        $suffix = $stocksInArchive[$code]
        $fileList += "$dateStr\$code.$suffix\*"
    }
    $fileList | Set-Content -Path $TEMP_LIST -Encoding ASCII

    # Extract to temp dir, keeping directory structure
    $tempExtractDir = "C:\Users\1\Desktop\temp_extract_$dateStr"
    New-Item -ItemType Directory -Force -Path $tempExtractDir | Out-Null

    $extractResult = & $SEVEN_ZIP x "$archivePath" "@$TEMP_LIST" -o"$tempExtractDir" -aoa -y 2>&1

    # Step 3: Move extracted files to per-stock directories
    $dateSubDir = Join-Path $tempExtractDir $dateStr
    $movedCount = 0
    if (Test-Path $dateSubDir) {
        $stockDirs = Get-ChildItem $dateSubDir -Directory -ErrorAction SilentlyContinue
        foreach ($sd in $stockDirs) {
            $codeMatch = [regex]::Match($sd.Name, '^(\d{6})\.(SZ|SH)$')
            if (-not $codeMatch.Success) { continue }
            $code = $codeMatch.Groups[1].Value
            if (-not $UNIVERSE.ContainsKey($code)) { continue }

            $targetDir = Join-Path $OUTPUT_ROOT "$code\raw\$dateStr"
            New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

            Get-ChildItem $sd.FullName -File -ErrorAction SilentlyContinue | ForEach-Object {
                Move-Item $_.FullName $targetDir -Force -ErrorAction SilentlyContinue
                $movedCount++
            }
        }
    }

    # Step 4: Clean temp, delete archive
    Remove-Item -Recurse -Force $tempExtractDir -ErrorAction SilentlyContinue
    Remove-Item $TEMP_LIST -ErrorAction SilentlyContinue

    if ($movedCount -gt 0) {
        Remove-Item $archivePath -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path $archivePath)) {
            $deleted++
            $freedGB += $archiveSizeGB
            Log "  -> $movedCount files ($nStocks stocks), DELETED archive (+$archiveSizeGB GB)"
        } else {
            $failed++
            Log "  -> $movedCount files, WARN: could not delete archive"
        }
    } else {
        $failed++
        Log "  -> FAIL: no files extracted"
    }
}

$stopwatch.Stop()
$totalMin = [math]::Round($stopwatch.Elapsed.TotalMinutes, 1)

Log ""
Log "========================================"
Log "DONE in ${totalMin}min"
Log "  Processed: $processed"
Log "  Deleted:   $deleted"
Log "  Failed:    $failed"
Log "  Freed:     $freedGB GB"
Log "========================================"
