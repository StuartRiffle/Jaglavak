// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#define CUDA_REQUIRE( _CALL ) \
{ \
    cudaError_t status = (_CALL); \
    if( status != cudaSuccess ) \
    { \
        printf( "ERROR: failure in " #_CALL " [%s]\n", cudaGetErrorName( status ) ); \
        return; \
    } \
}

template< typename T >
struct CudaBuffer
{
    T*      mHost;
    T*      mDevice;
    size_t  mOffset;
    size_t  mBufferSize;

    CudaBuffer() { this->SetNull(); }

    void SetNull() { mBufferSize = 0; }
    bool IsNull() const { return (mBufferSize != 0); }

    T& operator[]( size_t idx )
    {
        assert( idx * sizeof( T ) < mBufferSize );
        return mHost[idx];
    }

    void CopyUpToDeviceAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemcpyAsync( mDevice, mHost, mBufferSize, cudaMemcpyHostToDevice, stream );
    }

    void CopyDownToHostAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemcpyAsync( mHost, mDevice, mBufferSize, cudaMemcpyDeviceToHost, stream );
    }

    void ClearOnDeviceAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemsetAsync( mDevice, 0, mBufferSize, stream );
    }   
};

#include "Allocator.h"

class CudaAllocator
{
    void*   mHostBuffer;
    void*   mDeviceBuffer;
    size_t  mHeapSize;

    HeapAllocator< uintptr_t > mHeap;

public:

    CudaAllocator()
    {
        mHostBuffer = NULL;
        mDeviceBuffer = NULL;
        mHeapSize = 0;
    }

    ~CudaAllocator()
    {
        this->Shutdown();
    }

    void Init( size_t heapSize )
    {
        mHostBuffer = NULL;
        mDeviceBuffer = NULL;
        mHeapSize = heapSize;
        mHeap.Init( mHeapSize );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mHostBuffer, heapSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mDeviceBuffer, heapSize ) ));        
    }

    void Shutdown()
    {
        assert( mHostBuffer );
        CUDA_REQUIRE(( cudaFreeHost( mHostBuffer ) ));
        mHostBuffer = NULL;

        assert( mDeviceBuffer );
        CUDA_REQUIRE(( cudaFree( mDeviceBuffer ) ));        
        mDeviceBuffer = NULL;
    }

    template< typename T >
    void Alloc( size_t count, CudaBuffer< T >* dest )
    {
        size_t size = count * sizeof( T );
        size_t offset = mHeap.Alloc( size );

        dest->mHost   = (T*) ((uintptr_t) mHostBuffer   + offset);
        dest->mDevice = (T*) ((uintptr_t) mDeviceBuffer + offset);
        dest->mOffset = offset;
        dest->mBufferSize = size;                                 
    }

    template< typename T >
    void Free( CudaBuffer< T >& buf )
    {
        if( !buf.IsNull() )
        {
            mHeap.Free( buf.mOffset );
            buf.SetNull();
        }
    }
};



extern void PlayGamesCudaAsync( 
    const PlayoutParams* params, 
    const Position* pos, 
    ScoreCard* dest, 
    int count,
    int blockCount, 
    int blockSize, 
    cudaStream_t stream );

