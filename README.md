# Jaglavak
Jaglavak is a chess engine that uses Monte Carlo Tree Search.

## Building Jaglavak

### Windows



### Linux

Starting from a clean install of Ubuntu 18.04 LTS (ubuntu:latest)

1) Install the packages needed for building
```sudo apt update
sudo apt install -y build-essential git cmake nvidia-cuda-dev
```
2) Clone the latest version of Jaglavak
```git clone https://github.com/StuartRiffle/Jaglavak
```
3) Build it in the normal CMake way
```cd Jaglavak && mkdir Build && cd Build && cmake .. && make
```
4) Run to test
```./Jaglavak
```
powershell @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

choco install /y git cmake visualstudio2019buildtools cuda





sudo apt update
sudo apt install -y build-essential git cmake nvidia-cuda-dev
git clone https://github.com/StuartRiffle/Jaglavak
cd Jaglavak && mkdir Build && cd Build && cmake .. && make
./Jaglavak
