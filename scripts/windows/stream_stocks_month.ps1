param([string]$Month, [string[]]$Codes)
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
Set-Location $ss

$dirs = @()
foreach ($code in $Codes) {
    $rawDir = "$code\raw"
    if (-not (Test-Path $rawDir)) { continue }
    $dateDirs = Get-ChildItem $rawDir -Directory | Where-Object { $_.Name -like "$Month*" }
    foreach ($dd in $dateDirs) {
        $dirs += "$code/raw/$($dd.Name)"
    }
}

if ($dirs.Count -eq 0) { exit 0 }

$tempList = [System.IO.Path]::GetTempFileName()
$dirs -join "`n" | Out-File -FilePath $tempList -Encoding ASCII
& tar -cf - -T $tempList
Remove-Item $tempList -ErrorAction SilentlyContinue
