# Check disk usage on Windows C drive
$targets = @(
    "C:\Users\1\Desktop\2026",
    "C:\Users\1\Desktop\2025",
    "C:\Users\1\Desktop\institution-alpha",
    "C:\Users\1\Desktop\JARVIS_AgentOS_v2",
    "C:\Users\1\Desktop\_002516_all",
    "C:\Users\1\Desktop\_002516_extract",
    "C:\Users\1\Desktop\HR*",
    "C:\Users\1\AppData\Local\Temp",
    "C:\Windows\Temp"
)

Write-Host "=== C: Drive Space ==="
$c = Get-PSDrive C
$freeGB = [math]::Round($c.Free / 1GB, 2)
$usedGB = [math]::Round($c.Used / 1GB, 2)
Write-Host "Free: ${freeGB} GB / Used: ${usedGB} GB / Total: $([math]::Round(($c.Free+$c.Used)/1GB, 2)) GB"

Write-Host ""
Write-Host "=== Largest Directories on Desktop ==="
Get-ChildItem C:\Users\1\Desktop -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB, 1)
    Write-Host "$sizeMB MB  --  $($_.Name)"
} | Sort-Object -Descending

Write-Host ""
Write-Host "=== Temp Directories ==="
foreach ($d in @("C:\Users\1\AppData\Local\Temp", "C:\Windows\Temp", "C:\Temp")) {
    if (Test-Path $d) {
        $size = (Get-ChildItem $d -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 1)
        Write-Host "$d : $sizeMB MB"
    }
}

Write-Host ""
Write-Host "=== Largest Files on Desktop ==="
Get-ChildItem C:\Users\1\Desktop -File -ErrorAction SilentlyContinue | Sort-Object Length -Descending | Select-Object -First 10 | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "$sizeMB MB  --  $($_.Name)"
}
