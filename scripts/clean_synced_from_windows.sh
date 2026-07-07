#!/bin/bash
# Delete 202501 data from Windows for stocks that are already synced to Mac
MONTH="${1:-202501}"
DEST="/Users/zhonghuagang/Desktop/量化/institution-alpha/data/single_stock"

# Find all stock dirs on Mac that have $MONTH data
echo "Finding synced stocks on Mac..."
mac_stocks=$(find "$DEST" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | while read d; do
    code=$(basename "$d")
    count=$(find "$d/raw" -maxdepth 1 -name "${MONTH}*" -type d 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        echo "$code:$count"
    fi
done)

total_stocks=$(echo "$mac_stocks" | grep -c ":")
echo "Found $total_stocks stocks with ${MONTH} data on Mac"
echo "$mac_stocks" | head -5
echo "..."

# Delete each stock's $MONTH data from Windows
freed=0
count=0
while IFS=: read -r code dates; do
    [ -z "$code" ] && continue
    echo -n "[$count/$total_stocks] $code ($dates dirs)... "
    result=$(ssh win-train "powershell -Command \"Remove-Item -Recurse -Force 'C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\\$code\\raw\\${MONTH}*' -ErrorAction SilentlyContinue; if (-not (Test-Path 'C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\\$code\\raw\\${MONTH}*')) { Write-Host 'OK' } else { Write-Host 'FAIL' } \"" 2>/dev/null)
    echo "$result"
    if [ "$result" = "OK" ]; then
        count=$((count + 1))
    fi
done <<< "$mac_stocks"

echo ""
echo "Deleted $count stocks' ${MONTH} data from Windows"

# Check Windows disk
free=$(ssh win-train "powershell -Command \"[math]::Round((Get-PSDrive C).Free/1GB,1)\"" 2>/dev/null)
echo "Windows free: ${free} GB"
