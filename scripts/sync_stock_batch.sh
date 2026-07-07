#!/bin/bash
# Sync one stock's month data from Windows to Mac
# Usage: ./sync_stock_batch.sh <code> <month>

CODE="$1"
MONTH="$2"
DEST="/Users/zhonghuagang/Desktop/量化/institution-alpha/data/single_stock"

if [ -z "$CODE" ] || [ -z "$MONTH" ]; then
    echo "Usage: $0 <code> <month>"
    exit 1
fi

# Check if already fully synced on Mac
MAC_DIRS=$(find "$DEST/$CODE/raw" -maxdepth 1 -name "${MONTH}*" -type d 2>/dev/null | wc -l | tr -d ' ')
if [ "$MAC_DIRS" -gt 0 ]; then
    echo "[$CODE] Already has $MAC_DIRS ${MONTH} dirs on Mac, skipping"
    exit 0
fi

echo "[$CODE] Syncing ${MONTH}..."
mkdir -p "$DEST"

# Stream from Windows and extract
ssh win-train "powershell -File C:\\Users\\1\\Desktop\\stream_stock_month.ps1 -Code $CODE -Month $MONTH" 2>/dev/null | tar -xf - -C "$DEST" 2>/dev/null

if [ $? -eq 0 ]; then
    MAC_COUNT=$(find "$DEST/$CODE/raw" -maxdepth 1 -name "${MONTH}*" -type d 2>/dev/null | wc -l | tr -d ' ')
    echo "[$CODE] Synced: $MAC_COUNT date-dirs"

    # Delete from Windows to free space
    ssh win-train "powershell -Command \"Remove-Item -Recurse -Force 'C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\\$CODE\\raw\\${MONTH}*' -ErrorAction SilentlyContinue; Write-Host 'Deleted ${MONTH} data for $CODE'\"" 2>/dev/null
else
    echo "[$CODE] Sync failed"
    exit 1
fi
