# Delete HS300-only stock directories from single_stock
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$deleteList = "C:\Users\1\Desktop\hs300_delete.txt"

$codes = Get-Content $deleteList
$deleted = 0
$freedMB = 0

foreach ($c in $codes) {
    $dir = Join-Path $ss $c
    if (Test-Path $dir) {
        $size = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        if (-not (Test-Path $dir)) {
            $deleted++
            $freedMB += ($size / 1MB)
        }
    }
}

$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
$freedGB = [math]::Round($freedMB / 1024, 1)
$remaining = (Get-ChildItem $ss -Directory).Count
Write-Host "Deleted $deleted dirs, freed $freedGB GB, disk free: $freeGB GB, remaining: $remaining dirs"
