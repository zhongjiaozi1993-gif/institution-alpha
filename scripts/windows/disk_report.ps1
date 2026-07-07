Write-Host "=== C: Drive ==="
$c = Get-PSDrive C
$total = [math]::Round(($c.Used + $c.Free) / 1GB, 1)
Write-Host "Used: $([math]::Round($c.Used/1GB,1)) GB / Total: $total GB, Free: $([math]::Round($c.Free/1GB,1)) GB"

Write-Host ""
Write-Host "=== Desktop top-level ==="
Get-ChildItem "C:\Users\1\Desktop" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeGB = [math]::Round($size / 1GB, 1)
    Write-Host "  $sizeGB GB -- $($_.Name)"
}

Write-Host ""
Write-Host "=== 2025 .7z remaining ==="
Get-ChildItem "C:\Users\1\Desktop\2025" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $files = Get-ChildItem $_.FullName -Filter "*.7z" -ErrorAction SilentlyContinue
    if ($files.Count -gt 0) {
        $size = [math]::Round(($files | Measure-Object -Property Length -Sum).Sum / 1GB, 1)
        Write-Host "  $($_.Name): $($files.Count) .7z, $size GB"
    } else {
        Write-Host "  $($_.Name): 0 .7z (deleted)"
    }
}

Write-Host ""
Write-Host "=== single_stock ==="
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
if (Test-Path $ss) {
    $dirs = (Get-ChildItem $ss -Directory -ErrorAction SilentlyContinue).Count
    $size = [math]::Round((Get-ChildItem $ss -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB, 1)
    Write-Host "  $dirs dirs, $size GB"
}

Write-Host ""
Write-Host "=== institution-alpha top-level ==="
Get-ChildItem "C:\Users\1\Desktop\institution-alpha" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($size -gt 1MB) {
        $sizeGB = [math]::Round($size / 1GB, 1)
        Write-Host "  $sizeGB GB -- $($_.Name)"
    } else {
        Write-Host "  <1 MB -- $($_.Name)"
    }
}
