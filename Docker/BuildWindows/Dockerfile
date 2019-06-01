FROM mcr.microsoft.com/windows:1809

# Install Chocolatey
RUN @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" \
-NoProfile -InputFormat None -ExecutionPolicy Bypass \
-Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && \
SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

RUN choco install /y git
RUN choco install /y cuda
RUN choco install /y cmake --installargs 'ADD_CMAKE_TO_PATH=System'
RUN refreshenv

ADD https://aka.ms/vs/16/release/vs_buildtools.exe vs_buildtools.exe
RUN C:\vs_buildtools.exe --quiet --wait --norestart --nocache \
    --installPath C:\BuildTools \
    --all \
    --remove Microsoft.VisualStudio.Component.Windows10SDK.10240 \
    --remove Microsoft.VisualStudio.Component.Windows10SDK.10586 \
    --remove Microsoft.VisualStudio.Component.Windows10SDK.14393 \
    --remove Microsoft.VisualStudio.Component.Windows81SDK 

ENTRYPOINT C:\BuildTools\Common7\Tools\VsDevCmd.bat


