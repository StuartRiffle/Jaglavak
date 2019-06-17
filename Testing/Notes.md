# Jaglavak

- Jaglavak is a chess engine based on [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS).
- This is an **asynchronous** implementation of MCTS that allows for high throughput. 
- GPU f sdf daf 
- CPU playouts use **SIMD** hardware (up to 8-wide with AVX-512), and run on all cores.

- Jaglavak is not sensitive or smart. It is a brute-force firehose of chess games. 
- 

## ELI5

 is an algorithm for searching huge trees. Jaglavak uses MCTS to choose the best move.




## Technical Status

- CPU workers
OpenMP, moving to thread pool to adjust priority

- GPU workers
One per CUDA device. 

/*
Platform support
UCI protocol
Multicore vs multisite
### endgame tablebases
### opening book
### Three-fold repetition ignored
Tuning
MCTS
Brute force approach
MIT License
Time management
Threading

C++11 source
64 bit architectures
Uses bitboards
Interruptible multithread Perft test
Ponder?
No Evaluation function
Testing
Playing strength
SIMD scaling
GPU scaling
UCT
Branch free move generation

Branch

Technical details

- Board rappresentation QuadBitBoard

- Magic BitBoard

- Multithreaded search

- PV search

- Quiescence search with captures, promotions and checks

- Transposition Table

- Iterative deepening

- Internal iterative deepening

- Aspiration window

- Search extensions and reductions

- Singular extension

- Late move pruning

- Futility and Null Move pruning

- Delta pruning

- SEE

- History

- Killer moves

- Countermove Heuristic

- Razoring

- Skill levels

- MultiPV

- Chess960

- Syzygy egtb support

- Polyglot book support


Better time management
Opening book and tablebase support
*/
### POPCNT

[POPCNT](https://www.chessprogramming.org/Population_Count) is a CPU instruction for calculating how many bits in a number are 1, as opposed to 0.
In bitboard engines, every 1 bit represents a piece, so you can use POPCNT to quickly count the pieces of different types, what their potential targets are, how many squares are under player control, etc. 
It gives enough of a speedup for some chess engines that they make a special build for computers with POPCNT support.

Jaglavak doesn't use POPCNT much, so it doesn't benefit from a special build. POPCNT is used on the SIMD code paths, but the scalar code does not check.

### SIMD

The code in Jaglavak to detect valid moves (given a chess position), and the code to update the board when these moves are made, is branch-free.
That means that it doesn't need to make any decisions, so it just goes through the same motions every time, no matter what input feed it. 
Branch-free style code is ideal for a GPU, because all the threads can just follow the plan and do the same thing, together. So there is no divergence and the GPU can run on all cylinders. 

It's good for SIMD too, 





## Getting Started

Jaglavak is a console application, written in C++ with SIMD intrinsics.

Basically, there is a chess part and an MCTS part. 


## Linux setup

The Linux build uses [CMake 3.8+](https://cmake.org/download/). A few packages are required. To install them (on Ubuntu): 
                                      
    sudo apt update
    sudo apt install -y build-essential git cmake nvidia-cuda-dev
                                
(The same commands will work if you're using [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl)).

## Windows setup

Building Jaglavak natively on Windows requires:

- [Visual Studio 2019](https://visualstudio.microsoft.com/downloads)
- [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-downloads)

Open `Project\Jaglavak.sln`, build, and run.

The CMake method below also works on Windows.

## Building the code

Clone the latest version of Jaglavak:

    git clone https://github.com/StuartRiffle/Jaglavak

Build it the CMake way:

    cd Jaglavak/Build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

If everything went well, you will find the executable there in the Build folder. Type "Jaglavak" to run it.


## CPU support

Jaglavak was designed for parallel operation. The core code is branch-free, which allows multiple games of chess to be played at once using SIMD registers.

| Instruction set | SIMD | Speedup |
| --- | --- | --- |
| x64 | 1 | - |
| SSE4.1 | 2 | 1.8x |
| AVX2 | 4 | 4.2x |
| AVX-512 | 8 | 6.3x |

# GPU support

Jaglavak supports multiple CUDA devices, and will load balance between them. 
The same codebase is used for both CPU and GPU, and the branch-free style of the code maps well to CUDA hardware. 

_However_, the engine does pretty much everything using 64-bit integers, which are many times slower on GPU. 
Current generation devices only support 32-bit words, so 64-bit operations have to be emulated using multiple instructions. This wastes a lot of cycles.




# Code 

Chess 
origin

SIMD
branch-free
GPU

MCTS
MRU structure


Players


# Tradeoffs


Jaglavak takes a brute-force approach. It does not understand chess, beyond identifying legal moves. 

The rollout policy is completely random. Moves are made until one side wins, or the game becomes a known draw.

Goal: the inner loop should fit in L1 instruction cache




The 
- Focus on scalability, speed, and smartness (in that order)

- performance 


 # Status

 




## Description

sdfasdf



This is Carvallo's performance on a 1.9Ghz PC:

Test suite	Time per position	Version 1.7	Version 1.6	Version 1.5
WinAtChess (New)	1 second	293/300	293/300	291/300
SilentButDeadly	1 second	123/134	120/134	120/134
ECMGCP	1 second	110/183	101/183	86/183
ECMGCP	5 seconds	154/183	145/183	138/183
Arasan 19a	60 seconds	52/200	40/200	35/200
