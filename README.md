# Jaglavak
Jaglavak is a chess engine that uses Monte Carlo Tree Search.

## Building Jaglavak

### Windows

### Linux

Install the packages needed for building:
```apt update
apt install -y git cmake build-essential nvidia-cuda-dev
```
Clone the latest version of the engine:
```git clone https://github.com/StuartRiffle/Jaglavak
cd Jaglavak
```
Build it with CMake:
```mkdir build
cd build
cmake ..
make
```
And if all goes well, run it:
```./Jaglavak```

