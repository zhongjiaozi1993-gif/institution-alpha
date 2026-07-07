$desktop = "C:\Users\1\Desktop"
# Match by garbled name visible in SSH output
$target = Get-ChildItem $desktop -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '˫Ӱ' }

if (-not $target) {
    # Fallback: try matching by file count or size pattern (~114 GB)
    $all = Get-ChildItem $desktop -Directory -ErrorAction SilentlyContinue
    $known = @("2025", "2026", "2026_level2_ops", "auto-page-turner", "institution-alpha", "JARVIS_AgentOS_v2", "_002516_all", "_002516_extract")
    foreach ($d in $all) {
        if ($d.Name -notin $known) {
            $size = (Get-ChildItem $d.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
            $sizeGB = [math]::Round($size / 1GB, 1)
            Write-Host "Candidate: $($d.Name) = $sizeGB GB"
        }
    }
    Write-Host "No match found"
    exit 1
}

$path = $target.FullName
$size = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$sizeGB = [math]::Round($size / 1GB, 1)
Write-Host "Found: $($target.Name) ($sizeGB GB)"

Remove-Item -Recurse -Force $path -ErrorAction Stop
Write-Host "Deleted successfully"

$free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
Write-Host "Free: $free GB"
