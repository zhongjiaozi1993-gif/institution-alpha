# Test v2 batch extraction on a single archive with 2 stocks
param(
    [switch]$Cleanup
)

$ErrorActionPreference = "Continue"
$SEVEN_ZIP = "C:\Users\1\Desktop\7zr.exe"
$ARCHIVE = "C:\Users\1\Desktop\2025\202504\20250401.7z"
$TEMP_LIST = "C:\Users\1\Desktop\temp_test_list.txt"
$TEMP_DIR = "C:\Users\1\Desktop\temp_test_extract"
$OUTPUT_DIR = "C:\Users\1\Desktop\temp_test_output"
$DATE_STR = "20250401"

Write-Host "=== Test v2 Single Archive ==="

# List universe stocks in this archive
$UNIVERSE_FILE = "C:\Users\1\Desktop\institution-alpha\data\processed\stock_universe\index_universe.txt"
$UNIVERSE = @{}
Get-Content $UNIVERSE_FILE | Where-Object { $_ -match '^\d{6}$' } | ForEach-Object { $UNIVERSE[$_] = $true }

$listOutput = & $SEVEN_ZIP l "$ARCHIVE" -ba 2>&1
$stocksInArchive = @{}
foreach ($line in $listOutput) {
    if ($line -match '(\d{6})\.(SZ|SH)') {
        $code = $Matches[1]
        if ($UNIVERSE.ContainsKey($code)) {
            $stocksInArchive[$code] = $Matches[2]
        }
    }
}

Write-Host "Universe stocks in archive: $($stocksInArchive.Count)"
# Take just first 3 for test
$testCodes = @($stocksInArchive.Keys | Select-Object -First 3)
Write-Host "Testing with: $($testCodes -join ', ')"

# Build file list
$fileList = @()
foreach ($code in $testCodes) {
    $suffix = $stocksInArchive[$code]
    $fileList += "$DATE_STR\$code.$suffix\*"
}
$fileList | Set-Content -Path $TEMP_LIST -Encoding ASCII
Write-Host "File list:"
Get-Content $TEMP_LIST

# Extract
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null
Write-Host "Extracting..."
$extractResult = & $SEVEN_ZIP x "$ARCHIVE" "@$TEMP_LIST" "-o$TEMP_DIR" -aoa -y 2>&1
$lastLine = $extractResult | Select-Object -Last 5
Write-Host "Extract tail: $lastLine"

# Check what was extracted
Write-Host ""
Write-Host "Extracted structure:"
Get-ChildItem $TEMP_DIR -Recurse -File | ForEach-Object {
    $sizeKB = [math]::Round($_.Length / 1KB, 1)
    $relPath = $_.FullName.Replace($TEMP_DIR, "")
    Write-Host "  $sizeKB KB -- $relPath"
}

# Move to output (simulating the real move)
Write-Host ""
Write-Host "Moving files..."
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
$moved = 0
$dateSubDir = Join-Path $TEMP_DIR $DATE_STR
if (Test-Path $dateSubDir) {
    $stockDirs = Get-ChildItem $dateSubDir -Directory
    foreach ($sd in $stockDirs) {
        $targetDir = Join-Path $OUTPUT_DIR "$($sd.Name)\$DATE_STR"
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        Get-ChildItem $sd.FullName -File | ForEach-Object {
            Move-Item $_.FullName $targetDir -Force
            $moved++
        }
    }
}
Write-Host "Moved $moved files to $OUTPUT_DIR"

# Verify output
Write-Host ""
Write-Host "Output structure:"
Get-ChildItem $OUTPUT_DIR -Recurse -File | ForEach-Object {
    $sizeKB = [math]::Round($_.Length / 1KB, 1)
    $relPath = $_.FullName.Replace($OUTPUT_DIR, "")
    Write-Host "  $sizeKB KB -- $relPath"
}

if ($Cleanup) {
    Write-Host ""
    Write-Host "Cleaning up..."
    Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
    Remove-Item $TEMP_LIST -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $OUTPUT_DIR -ErrorAction SilentlyContinue
    Write-Host "Done"
}
