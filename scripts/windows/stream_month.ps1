param([string]$Month)
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
Set-Location $ss

# Collect directory paths (no Chinese chars needed in paths we output)
$dirs = @()
Get-ChildItem . -Directory | ForEach-Object {
    $code = $_.Name
    $rawDir = Join-Path $code "raw"
    if (-not (Test-Path $rawDir)) { return }
    $dateDirs = Get-ChildItem $rawDir -Directory | Where-Object { $_.Name -like "$Month*" }
    foreach ($dd in $dateDirs) {
        $dirs += "$code/raw/$($dd.Name)"
    }
}

if ($dirs.Count -eq 0) {
    Write-Error "No dirs found for $Month"
    exit 1
}

# Write dir list to temp file then feed to tar
$tempList = [System.IO.Path]::GetTempFileName()
$dirs -join "`n" | Out-File -FilePath $tempList -Encoding ASCII

& tar -cf - -T $tempList
Remove-Item $tempList -ErrorAction SilentlyContinue
