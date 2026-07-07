$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$totalStocks = (Get-ChildItem $ss -Directory).Count

$dates = @("20250528", "20250529", "20250530", "20250626", "20250627", "20250630")
$normalDates = @("20250527", "20250625")

foreach ($d in $dates) {
    $count = 0
    Get-ChildItem $ss -Directory | ForEach-Object {
        if (Test-Path (Join-Path $_.FullName "raw\$d")) { $count++ }
    }
    Write-Host "$d : $count / $totalStocks stocks"
}

Write-Host "--- Normal dates for comparison ---"
foreach ($d in $normalDates) {
    $count = 0
    Get-ChildItem $ss -Directory | ForEach-Object {
        if (Test-Path (Join-Path $_.FullName "raw\$d")) { $count++ }
    }
    Write-Host "$d : $count / $totalStocks stocks"
}
