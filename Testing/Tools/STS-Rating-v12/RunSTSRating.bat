:: A. UCI engines
::============

:: Run to get sts rating using --getrating option
:: when --getrating is used, num threads will be set to 1, and movetime will also be
:: set by the tool depending on the speed of your machine. On my machine without other loads
:: the movetime per pos used by the tool is 200ms. The tool will run a short benchmark
:: to measure your machine speed to get the movetime, before starting the test suite.


:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "Deuterium v2015.1.35.251.exe" -h 128 --proto uci --getrating --log

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "arasan18-64-popcnt.exe" -h 128 --proto uci --getrating --log

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "gaviota-1.0-win64-general.exe" -h 128 --proto uci --getrating

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "Laser-0_1-NOPOPCNT.exe" -h 128 --getrating --log

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "Houdini_4_x64B.exe" -t 1 -h 128 --getrating



:: Normal run to get score, movetime is in millisec

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "Stockfish 6.exe" -t 1 -h 128 --movetime 200




:: B. WINBOARD engines
:: ================

:: For winboard engines, sts rating is not applicable at the moment, but engine can be tested


:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "Deuterium v14.1.32.119.book.extract.exe" --proto wb --mps 40 --tc 0:8

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "LambChop_1099.exe" --proto wb --mps 300 --tc 1 --log

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "Gerbil_02_x64_ja.exe" --proto wb --mps 40 --tc 0:8 --log

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "Averno081.exe" --proto wb --mps 300 --tc 1 --log

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "Myrddin-087-64.exe" --proto wb --mps 40 --tc 0:8 --log

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "crafty-24.1-x64-sse42.exe" --proto wb --st 0.2 --san --log

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "EXchess_v7.71b_win64.exe" --proto wb --mps 40 --tc 0:8 --log --san

:: STS_Rating_v12 -f "STS1-STS15_LAN_v3.epd" -e "djinn1021_x64_popcnt.exe" --proto wb --mps 40 --tc 0:8 --log --san


