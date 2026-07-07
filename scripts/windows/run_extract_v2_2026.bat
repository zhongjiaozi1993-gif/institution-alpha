@echo off
cd /d C:\Users\1\Desktop
set LOG=C:\Users\1\Desktop\extract_v2_2026.log
echo [%date% %time%] START v2 extraction 2026 >> "%LOG%"
powershell -File C:\Users\1\Desktop\extract_universe_v2.ps1 -Year 2026 >> "%LOG%" 2>&1
echo [%date% %time%] DONE (exit=%ERRORLEVEL%) >> "%LOG%"
