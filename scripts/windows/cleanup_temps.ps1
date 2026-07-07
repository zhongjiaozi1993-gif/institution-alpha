# Cleanup temp directories on Windows to free space for DBSCAN pipeline
# Only deletes temporary/extracted data, never raw archives

$ErrorActionPreference = "Continue"
$root = "C:\Users\1\Desktop\institution-alpha\data"

$toClean = @(
    "$root\tmp_extract",
    "$root\tmp_archive_extract",
    "$root\tmp_sofia_v4_peers_full",
    "$root\tmp_extract_20250102_000001",
    "$root\tmp_peer_extract"
)

# Add all tmp_sofia_v4_peers_* dirs
$tmpDirs = Get-ChildItem $root -Directory -Filter "tmp_sofia_v4_peers_*" -ErrorAction SilentlyContinue
foreach ($d in $tmpDirs) {
    $toClean += $d.FullName
}

# Add all 0-byte sofia_v4_full_extract_* dirs
$emptyDirs = Get-ChildItem "$root\processed" -Directory -Filter "sofia_v4_full_extract_*" -ErrorAction SilentlyContinue
foreach ($d in $emptyDirs) {
    $size = (Get-ChildItem $d.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($size -eq 0) {
        $toClean += $d.FullName
    }
}

# Add 0-byte peer_ops_* dirs
$peerDirs = Get-ChildItem "$root\processed" -Directory -Filter "peer_ops_*" -ErrorAction SilentlyContinue
foreach ($d in $peerDirs) {
    $size = (Get-ChildItem $d.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($size -eq 0) {
        $toClean += $d.FullName
    }
}

# Add 0-byte baseline_* dirs
$baselineDirs = Get-ChildItem "$root\processed" -Directory -Filter "baseline_*" -ErrorAction SilentlyContinue
foreach ($d in $baselineDirs) {
    $size = (Get-ChildItem $d.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($size -eq 0) {
        $toClean += $d.FullName
    }
}

# Add level2_ops_YYYYMMDD checkpoint dirs (already output as CSV)
$opsCheckpointDirs = Get-ChildItem "$root\processed" -Directory -Filter "level2_ops_20*" -ErrorAction SilentlyContinue
foreach ($d in $opsCheckpointDirs) {
    $toClean += $d.FullName
}

Write-Host "=== Directories to clean ==="
$totalFreed = 0
foreach ($dir in $toClean) {
    if (Test-Path $dir) {
        $size = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Host "  $sizeMB MB  --  $dir"
        $totalFreed += $size

        # Actually delete
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        if ($?) {
            Write-Host "    -> DELETED"
        } else {
            Write-Host "    -> FAILED to delete"
        }
    } else {
        Write-Host "  (missing) -- $dir"
    }
}

$totalFreedGB = [math]::Round($totalFreed / 1GB, 2)
Write-Host ""
Write-Host "Total freed: $totalFreedGB GB"

# Check disk after
$c = Get-PSDrive C
$freeGB = [math]::Round($c.Free / 1GB, 2)
Write-Host "C: free after cleanup: $freeGB GB"
