#!/bin/bash
# Sync remaining 202501 data stock-by-stock, deleting from Windows after each success
MONTH="202501"
DEST="/Users/zhonghuagang/Desktop/量化/institution-alpha/data/single_stock"
mkdir -p "$DEST"

# Get list of Windows stocks that have $MONTH data
echo "Getting stock list from Windows..."
win_stocks=$(ssh win-train 'powershell -Command "
\$ss = \"C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\"
Get-ChildItem \$ss -Directory | ForEach-Object {
    \$raw = Join-Path \$_.FullName \"raw\"
    if ((Test-Path \$raw) -and (Get-ChildItem \$raw -Directory | Where-Object { \$_.Name -like \"'$MONTH'*\" } | Select-Object -First 1)) {
        Write-Output \$_.Name
    }
}
"' 2>/dev/null)

total=$(echo "$win_stocks" | grep -c "^[0-9]")
echo "Stocks with ${MONTH} data on Windows: $total"

synced=0
skipped=0
freed_mb=0
errors=0

while IFS= read -r code; do
    [ -z "$code" ] && continue
    [[ ! "$code" =~ ^[0-9]{6}$ ]] && continue

    # Check if already on Mac with all dates (17+ dirs)
    mac_count=$(find "$DEST/$code/raw" -maxdepth 1 -name "${MONTH}*" -type d 2>/dev/null | wc -l | tr -d ' ')

    if [ "$mac_count" -ge 17 ]; then
        # Already fully synced - delete from Windows if not already done
        result=$(ssh win-train "powershell -Command \"if (Test-Path 'C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\\$code\\raw\\${MONTH}*') { Remove-Item -Recurse -Force 'C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\\$code\\raw\\${MONTH}*' -ErrorAction SilentlyContinue; Write-Host 'deleted' } else { Write-Host 'already_gone' }\"" 2>/dev/null)
        skipped=$((skipped + 1))
        [ $((skipped % 50)) -eq 0 ] && echo "  Skipped $skipped already-synced stocks so far..."
        continue
    fi

    # Sync this stock
    echo -n "[$((synced + skipped + 1))/$total] $code... "

    # Stream from Windows and extract on Mac
    ssh win-train "powershell -File C:\\Users\\1\\Desktop\\stream_stock_month.ps1 -Code $code -Month $MONTH" 2>/dev/null | tar -xf - -C "$DEST" 2>/dev/null

    if [ $? -eq 0 ]; then
        # Verify
        new_count=$(find "$DEST/$code/raw" -maxdepth 1 -name "${MONTH}*" -type d 2>/dev/null | wc -l | tr -d ' ')

        if [ "$new_count" -gt 0 ]; then
            # Delete from Windows
            ssh win-train "powershell -Command \"Remove-Item -Recurse -Force 'C:\\Users\\1\\Desktop\\institution-alpha\\data\\single_stock\\$code\\raw\\${MONTH}*' -ErrorAction SilentlyContinue; Write-Host 'ok'\"" 2>/dev/null >/dev/null
            synced=$((synced + 1))
            echo "OK ($new_count dates)"
        else
            echo "FAIL (no dates extracted)"
            errors=$((errors + 1))
        fi
    else
        echo "FAIL (transfer error)"
        errors=$((errors + 1))
    fi

    # Progress report every 20 stocks
    if [ $(( (synced + skipped + errors) % 20 )) -eq 0 ]; then
        free_gb=$(ssh win-train "powershell -Command \"[math]::Round((Get-PSDrive C).Free/1GB,1)\"" 2>/dev/null)
        echo "  == Progress: $synced synced, $skipped skipped, $errors errors | Windows free: ${free_gb}GB =="
    fi

done <<< "$win_stocks"

echo ""
echo "============================================"
echo "SYNC COMPLETE"
echo "  Synced:  $synced"
echo "  Skipped: $skipped (already on Mac)"
echo "  Errors:  $errors"
free_gb=$(ssh win-train "powershell -Command \"[math]::Round((Get-PSDrive C).Free/1GB,1)\"" 2>/dev/null)
echo "  Windows free: ${free_gb} GB"
echo "============================================"
