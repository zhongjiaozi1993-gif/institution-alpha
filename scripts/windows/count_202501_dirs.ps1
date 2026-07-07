$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$count = 0
Get-ChildItem $ss -Directory | ForEach-Object {
    $raw = Join-Path $_.FullName "raw"
    if (Test-Path $raw) {
        $count += (Get-ChildItem $raw -Directory | Where-Object { $_.Name -like "202501*" }).Count
    }
}
Write-Host "Total 202501 date-dirs: $count"
