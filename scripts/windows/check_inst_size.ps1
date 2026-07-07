# Check institution-alpha directory sizes
$root = "C:\Users\1\Desktop\institution-alpha"

Write-Host "=== institution-alpha top-level ==="
Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB, 1)
    Write-Host "$sizeMB MB  --  $($_.Name)"
}

Write-Host ""
Write-Host "=== data subdirectories ==="
$dataDir = Join-Path $root "data"
if (Test-Path $dataDir) {
    Get-ChildItem $dataDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Host "$sizeMB MB  --  data/$($_.Name)"
    }
}

Write-Host ""
Write-Host "=== data/processed subdirectories ==="
$procDir = Join-Path $dataDir "processed"
if (Test-Path $procDir) {
    Get-ChildItem $procDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Host "$sizeMB MB  --  $($_.Name)"
    }
}

Write-Host ""
Write-Host "=== data/single_stock count ==="
$ssDir = Join-Path $dataDir "single_stock"
if (Test-Path $ssDir) {
    $count = (Get-ChildItem $ssDir -Directory -ErrorAction SilentlyContinue).Count
    $size = (Get-ChildItem $ssDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB, 1)
    Write-Host "single_stock: $count dirs, $sizeMB MB"
}
