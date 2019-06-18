// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

extern void PlayGamesCudaAsync( 
    const PlayoutParams* params, 
    const Position* pos, 
    ScoreCard* dest, 
    int count,
    int blockCount, 
    int blockSize, 
    cudaStream_t stream );
