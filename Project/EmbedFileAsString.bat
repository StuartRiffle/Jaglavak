@echo off
echo // JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
echo // GENERATED from %1 - DO NOT EDIT
echo.
echo namespace Embedded { const char* %~n1 = R"EMBEDDED_TEXT_FILE(
type %1
echo )EMBEDDED_TEXT_FILE"; };
