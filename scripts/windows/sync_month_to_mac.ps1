param([string]$Month)
# Lists all files under single_stock/{code}/raw/{Month}* with relative paths, one per line
$ss = "C:\Users\1\Desktop\institution-alpha\data\single_stock"
Get-ChildItem $ss -Directory | ForEach-Object {
    $code = $_.Name
    $rawDir = Join-Path $_.FullName "raw"
    if (-not (Test-Path $rawDir)) { return }
    $dateDirs = Get-ChildItem $rawDir -Directory | Where-Object { $_.Name -like "$Month*" }
    foreach ($dd in $dateDirs) {
        Get-ChildItem $dd.FullName -File | ForEach-Object {
            # Output relative path: {code}/raw/{date}/{filename}
            Write-Output "$code/raw/$($dd.Name)/$($_.Name)"
        }
    }
}
