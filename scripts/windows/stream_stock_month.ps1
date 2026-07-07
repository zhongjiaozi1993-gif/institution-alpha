param([string]$Code, [string]$Month)
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
Set-Location $ss

$rawDir = "$Code\raw"
if (-not (Test-Path $rawDir)) { exit 1 }

$dateDirs = Get-ChildItem $rawDir -Directory | Where-Object { $_.Name -like "$Month*" }
if ($dateDirs.Count -eq 0) { exit 0 }

$dirs = @()
foreach ($dd in $dateDirs) {
    $dirs += "$Code/raw/$($dd.Name)"
}

$tempList = [System.IO.Path]::GetTempFileName()
$dirs -join "`n" | Out-File -FilePath $tempList -Encoding ASCII
& tar -cf - -T $tempList
Remove-Item $tempList -ErrorAction SilentlyContinue
