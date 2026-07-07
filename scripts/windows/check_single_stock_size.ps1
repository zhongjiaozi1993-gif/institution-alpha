$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$total = [math]::Round((Get-ChildItem $ss -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB, 1)
$dirs = (Get-ChildItem $ss -Directory).Count
Write-Host "single_stock: $total GB, $dirs dirs"

# Check per-month sizes by scanning a few sample stocks
$samples = Get-ChildItem $ss -Directory | Select-Object -First 10

$monthSizes = @{}
foreach ($s in $samples) {
    $rawDir = Join-Path $s.FullName "raw"
    if (-not (Test-Path $rawDir)) { continue }
    Get-ChildItem $rawDir -Directory | ForEach-Object {
        $dateStr = $_.Name
        $month = $dateStr.Substring(0, 6)
        $size = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        if (-not $monthSizes.ContainsKey($month)) {
            $monthSizes[$month] = @()
        }
        $monthSizes[$month] += $size
    }
}

Write-Host "`nPer-month avg per stock (from $($samples.Count) samples):"
foreach ($month in ($monthSizes.Keys | Sort-Object)) {
    $avgMB = [math]::Round(($monthSizes[$month] | Measure-Object -Average).Average / 1MB, 1)
    $estTotalGB = [math]::Round($avgMB * $dirs / 1024, 1)
    $days = $monthSizes[$month].Count
    $avgPerDayMB = [math]::Round($avgMB / $days, 1)
    Write-Host "  $month : avg $avgMB MB/stock ($days days) -> est $estTotalGB GB total, ~$avgPerDayMB MB/stock/day"
}
