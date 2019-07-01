@echo off
setlocal
set VARNAME="%2"
if %VARNAME% EQU "" set VARNAME=%~n1
echo // JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
echo // GENERATED CODE - DO NOT EDIT THIS
echo.
echo const char* Embedded_%VARNAME% = R"EMBEDDED_FILE(
type %1
echo )EMBEDDED_FILE";

