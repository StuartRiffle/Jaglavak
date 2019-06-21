@echo off
echo Embedding settings
call EmbedFile ..\Settings\DefaultSettings.json > %TEMP%\DefaultSettings.h
fc %TEMP%\DefaultSettings.h DefaultSettings.h > NULL
if ERRORLEVEL 1 copy /y %TEMP%\DefaultSettings.h DefaultSettings.h
del %TEMP%\DefaultSettings.h
                                    
