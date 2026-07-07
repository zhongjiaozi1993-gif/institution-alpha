$months = @("202508","202509","202510","202511","202512")
foreach ($m in $months) {
    $dir = "C:\Users\1\Desktop\2025\$m"
    if (Test-Path $dir) {
        $files = Get-ChildItem $dir -Filter "*.7z" -ErrorAction SilentlyContinue
        $totalGB = [math]::Round(($files | Measure-Object -Property Length -Sum).Sum / 1GB, 1)
        Write-Host "$m : $($files.Count) .7z, $totalGB GB"
    }
}
$c = Get-PSDrive C
Write-Host "Free: $([math]::Round($c.Free/1GB,1)) GB"
