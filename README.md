# Jaglavak

- This is a chess engine based on [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS).
- It's an **asynchronous** implementation that allows for high throughput.
- It runs on GPU using all attached **CUDA** devices.
- CPU playouts run in **SIMD**, up to 8-wide.

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
### Three-fold repetition
Tuning
MCTS
Brute force approach
License

C++11 source
Bitboards
Interruptible multithread Perft test
64 bit architectures
Ponder?
No Evaluation function
Tuning method
Testing method



[ ] Cross-platform clean
[ ] 


### Testing
[ ] Set up CI
[X] Scripted positional "best move" testing
[ ] CI for ELO estimation
[ ] PERFT


### Chess
[X] 
[X] Fifty-move rule
[ ] Threefold repetition
[ ] Chess 960 (FRC)
 
### Optimization
[X] Use all cores
[X] Minimal small-block allocator
[X] Huge page support
[X] Batch playouts to run on CPU
[X] Batch _batches_ of playouts to run on GPU
[X] Branch-free game update
[X] Branch-free move generation
[ ] Branch-free move _selection_
[X] SIMD
    [X] x2 (SSE4.1)
    [X] x4 (AVX2)
    [X] x8 (AVX-512)
[X] GPU
    [X] CUDA
    [ ] OpenCL 2.0
    

[X] MCTS
[X] Multiple playouts per leaf
[X] Asynchronous playouts
[X] Memory limiting, node recycling
[ ] Prior probabilities for moves
[ ] 

[ ] 
[ ] Multi-threaded tree search
[ ] Rate-manage the queues to minimize latency (and prevent CUDA TDR)
[ ] Use NN inference (?) to generate priors for UCT
[ ] Local clustering, all nodes working on the same tree
[ ] Remote fooo, combining trees from different sites

### Parallel execution

### Meta
[X] UCI support
[ ] Pondering
[ ] Time management



[ ] 


*/
### POPCNT

[POPCNT](https://www.chessprogramming.org/Population_Count) is a CPU instruction for calculating how many bits in a number are 1, and not 0.
In bitboard engines, every 1 bit represents a piece, so you can use POPCNT to quickly count the pieces of different types, what their potential targets are, how many squares are under player control, etc. 
It gives enough of a speedup for some chess engines that they make a special build for computers with POPCNT support.

Jaglavak doesn't use POPCNT much, so it doesn't benefit from a special build. POPCNT is used on the SIMD code paths, but the scalar code does not check.

### Branch-Free Code

The code in Jaglavak to detect valid moves (given a chess position), and the code to update the board when these moves are made, is branch-free.
That means that it doesn't need to make any decisions, so it just goes through the same motions every time, no matter what input you feed it. 
Branch-free style code is ideal for a GPU, because all the threads can just follow the plan and do the same thing, together. So there is no divergence and the GPU can run on all cylinders. 


## Getting Started

Jaglavak is a chess engine that speaks the UCI protocol. It's a console application, written in C++ with SIMD intrinsics.
A 64-bit CPU is required




Basically, there is a chess part and an MCTS part. 


## Linux requirements

The Linux build uses [CMake 3.8+](https://cmake.org/download/). A few other packages are also required. To install them (on Ubuntu): 
                                      
    sudo apt update
    sudo apt install -y build-essential git cmake nvidia-cuda-dev
                                
(The same commands will work if you're using [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl)).

## Windows requirements

- [Visual Studio 2019](https://visualstudio.microsoft.com/downloads)
- [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-downloads)
- [CMake](https://chocolatey.org/packages/cmake.install)
- [git](https://chocolatey.org/packages/git.install)

## Building the code

Clone the latest version of Jaglavak:

    git clone https://github.com/StuartRiffle/Jaglavak

Build it in the CMake way:

    cd Jaglavak/Build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

If everything went well, you will find a new executable "Javlavak" there in the Build folder.

## Debugging

I normally debug using Visual Studio, 

Open `Project\Jaglavak.sln` and press `F5`. It will totally work.



## SIMD support

Jaglavak was designed for parallel operation. Many games of chess can be played at once using SIMD registers.

| Instruction set | SIMD | Speedup |
| --- | --- | --- |
| x64 | 1 | - |
| SSE4.1 | 2 | 1.8x |
| AVX2 | 4 | 4.2x |
| AVX-512 | 8 | 6.3x |

# GPU support

Jaglavak supports multiple CUDA devices, and load-balances between them (OpenCL 2 support is on the roadmap).
The same code is used for CPU and GPU, so the results are  way. 

The branch-free chess code maps well to compute hardware. But it does everything using 64-bit integers, which are _really_ slow on GPU. 

Current generation devices only support 32-bit words, so 64-bit operations have to be emulated using multiple instructions, and this wastes a lot of cycles right now. It should get better as the hardware catches up.

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


libboost_fiber-vc142-mt-gd-x64-1_70.lib
