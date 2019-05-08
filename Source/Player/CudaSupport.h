// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#define CUDA_REQUIRE( _CALL ) \
{ \
    cudaError_t status = (_CALL); \
    if( status != cudaSuccess ) \
    { \
        printf( "ERROR: failure in " #_CALL " [%d]\n", status ); \
        return; \
    } \
}

template< typename T >
struct CudaBuffer
{
    T* mHost;
    T* mDevice;
    size_t mBufferSize;

    CudaBuffer() : mHost( NULL ), mDevice( NULL ), mBufferSize( 0 ) {}

    ~CudaBuffer()
    {
        if( mHost )
            cudaFreeHost( mHost );

        if( mDevice )
            cudaFree( mDevice );
    }

    const T& operator[]( size_t idx ) const
    {
        return mHost[idx];
    }

    void Init( int count )
    {
        mHost = NULL;
        mDevice = NULL;
        mBufferSize = count * sizeof( T );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mHost, mBufferSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mDevice, mBufferSize ) ));        
    }
    
    void CopyToDeviceAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemcpyAsync( mDevice, mHost, mBufferSize, cudaMemcpyHostToDevice, stream );
    }

    void CopyToHostAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemcpyAsync( mHost, mDevice, mBufferSize, cudaMemcpyDeviceToHost, stream );
    }

    void ClearOnDeviceAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemsetAsync( mDevice, 0, mBufferSize, stream );
    }   
};

struct CudaLaunchSlot
{
    BatchRef                    mBatch;
    CudaBuffer< PlayoutParams > mParams;
    CudaBuffer< Position >      mInputs;
    CudaBuffer< ScoreCard >     mOutputs;
    int                         mCount;
    cudaEvent_t                 mReadyEvent; 
};

