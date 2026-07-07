#!/bin/bash
MONTH="202501"
DEST="/Users/zhonghuagang/Desktop/量化/institution-alpha/data/single_stock"

# Get list of synced stocks as array
mac_stocks=($(find "$DEST" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | while read d; do
    code=$(basename "$d")
    count=$(find "$d/raw" -maxdepth 1 -name "${MONTH}*" -type d 2>/dev/null | wc -l | tr -d ' ')
    [ "$count" -gt 0 ] && echo "$code"
done))

total=${#mac_stocks[@]}
echo "Synced stocks on Mac: $total"
echo "Deleting ${MONTH} data from Windows..."

freed=0
for ((i=0; i<total; i++)); do
    code="${mac_stocks[$i]}"
    [ -z "$code" ] && continue
    result=$(ssh win-train "powershell -Command \"Remove-Item -Recurse -Force 'C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\\$code\\raw\\${MONTH}*' -ErrorAction SilentlyContinue; if (Test-Path 'C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\\$code\\raw\\${MONTH}*') { Write-Host 'FAIL' } else { Write-Host 'OK' } \"" 2>/dev/null)
    echo "[$((i+1))/$total] $code: $result"
    freed=$((freed + 1))
done

echo "Deleted $freed stocks"
free=$(ssh win-train "powershell -Command \"[math]::Round((Get-PSDrive C).Free/1GB,1)\"" 2>/dev/null)
echo "Windows free: ${free} GB"
