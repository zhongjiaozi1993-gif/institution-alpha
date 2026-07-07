$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
$sample = (Get-ChildItem $ss -Directory | Select-Object -First 1).Name
Write-Host "Sample: $sample"

$dates = Get-ChildItem "$ss\$sample\raw" -Directory | Sort-Object Name
Write-Host "First: $($dates[0].Name)  Last: $($dates[-1].Name)  Total: $($dates.Count)"

Write-Host "---202505---"
$dates | Where-Object { $_.Name -like "202505*" } | ForEach-Object { Write-Host $_.Name }

Write-Host "---202506---"
$dates | Where-Object { $_.Name -like "202506*" } | ForEach-Object { Write-Host $_.Name }
