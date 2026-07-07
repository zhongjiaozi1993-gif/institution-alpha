param([string]$Month)
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$count = 0
Get-ChildItem $ss -Directory | ForEach-Object {
    $raw = Join-Path $_.FullName "raw"
    if (Test-Path $raw) {
        $found = Get-ChildItem $raw -Directory | Where-Object { $_.Name -like "$Month*" } | Select-Object -First 1
        if ($found) { $count++ }
    }
}
Write-Host $count
