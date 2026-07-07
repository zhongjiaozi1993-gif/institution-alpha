param([string]$Month)
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$count = 0
$totalMB = 0
Get-ChildItem $ss -Directory | ForEach-Object {
    $rawDir = Join-Path $_.FullName "raw"
    if (-not (Test-Path $rawDir)) { return }
    $dateDirs = Get-ChildItem $rawDir -Directory | Where-Object { $_.Name -like "$Month*" }
    foreach ($dd in $dateDirs) {
        $files = Get-ChildItem $dd.FullName -File
        $count += $files.Count
        $totalMB += ($files | Measure-Object -Property Length -Sum).Sum / 1MB
    }
}
$stocks = (Get-ChildItem "$ss" -Directory).Count
Write-Host "Month $Month : $count files, $([math]::Round($totalMB/1024,1)) GB, across $stocks stocks"
