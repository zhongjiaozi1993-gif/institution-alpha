param([string]$Month)
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$deleted = 0
$freedMB = 0

Get-ChildItem $ss -Directory | ForEach-Object {
    $rawDir = Join-Path $_.FullName "raw"
    if (-not (Test-Path $rawDir)) { return }
    $dateDirs = Get-ChildItem $rawDir -Directory | Where-Object { $_.Name -like "$Month*" }
    foreach ($dd in $dateDirs) {
        $size = (Get-ChildItem $dd.FullName -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        Remove-Item -Recurse -Force $dd.FullName -ErrorAction SilentlyContinue
        if (-not (Test-Path $dd.FullName)) {
            $deleted++
            $freedMB += ($size / 1MB)
        }
    }
}

$freedGB = [math]::Round($freedMB / 1024, 1)
$free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
Write-Host "Deleted $deleted date-dirs for $Month, freed $freedGB GB, free: $free GB"
