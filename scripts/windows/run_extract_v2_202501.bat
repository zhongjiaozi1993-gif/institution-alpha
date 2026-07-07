@echo off
cd /d C:\Users\1\Desktop
powershell -File C:\Users\1\Desktop\extract_universe_v2.ps1 -Year 2025 -Month 01 -LogFile "C:\Users\1\Desktop\extract_v2_202501.log"
echo [%date% %time%] DONE (exit=%ERRORLEVEL%) >> "C:\Users\1\Desktop\extract_v2_202501.log"
