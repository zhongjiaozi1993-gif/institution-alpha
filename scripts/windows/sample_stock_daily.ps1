$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$stock = (Get-ChildItem $ss -Directory | Select-Object -First 1).Name
Write-Host "Stock: $stock"
$raw = "$ss\$stock\raw"
Get-ChildItem $raw -Directory | Sort-Object Name | ForEach-Object {
    $size = [math]::Round((Get-ChildItem $_.FullName -File | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Host "$($_.Name): $size MB"
}
