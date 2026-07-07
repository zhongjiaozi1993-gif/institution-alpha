param([string]$Code)
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
Set-Location $ss
$rawDir = "$Code\raw"
if (-not (Test-Path $rawDir)) { exit 1 }
Get-ChildItem $rawDir -Directory | ForEach-Object {
    Get-ChildItem $_.FullName -File | ForEach-Object {
        Write-Output "$Code/raw/$($dd.Name)/$($_.Name)"
    }
}
