param([string]$ListFile, [string]$TargetDir)
$codes = Get-Content $ListFile
$deleted = 0; $freedMB = 0
foreach ($c in $codes) {
    $dir = Join-Path $TargetDir $c
    if (Test-Path $dir) {
        $size = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        if (-not (Test-Path $dir)) { $deleted++; $freedMB += ($size / 1MB) }
    }
}
$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
$freedGB = [math]::Round($freedMB / 1024, 1)
$remaining = (Get-ChildItem $TargetDir -Directory).Count
Write-Host "Deleted $deleted dirs, freed $freedGB GB, free: $freeGB GB, remaining: $remaining"
