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
    T*      _Host;
    T*      _Device;
    size_t  _Offset;
    size_t  _BufferSize;

    CudaBuffer() { this->SetNull(); }

    void SetNull() { _BufferSize = 0; }
    bool IsNull() const { return (_BufferSize == 0); }

    T& operator[]( size_t idx )
    {
        assert( idx * sizeof( T ) < _BufferSize );
        return _Host[idx];
    }

    void CopyUpToDeviceAsync( cudaStream_t stream = NULL ) const
    {
        CUDA_REQUIRE(( cudaMemcpyAsync( _Device, _Host, _BufferSize, cudaMemcpyHostToDevice, stream ) ));
    }

    void CopyDownToHostAsync( cudaStream_t stream = NULL ) const
    {
        CUDA_REQUIRE(( cudaMemcpyAsync( _Host, _Device, _BufferSize, cudaMemcpyDeviceToHost, stream ) ));;
    }

    void ClearOnDeviceAsync( cudaStream_t stream = NULL ) const
    {
        CUDA_REQUIRE(( cudaMemsetAsync( _Device, 0, _BufferSize, stream ) ));
    }   
};

#include "Allocator.h"

class CudaAllocator
{
    void*   _HostBuffer;
    void*   _DeviceBuffer;
    size_t  _HeapSize;

    HeapAllocator _Heap;

public:

    CudaAllocator()
    {
        _HostBuffer = NULL;
        _DeviceBuffer = NULL;
        _HeapSize = 0;
    }

    ~CudaAllocator()
    {
        this->Shutdown();
    }

    void Init( size_t heapSize )
    {
        _HostBuffer = NULL;
        _DeviceBuffer = NULL;
        _HeapSize = heapSize;
        _Heap.Init( _HeapSize );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &_HostBuffer, heapSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &_DeviceBuffer, heapSize ) ));        
    }

    void Shutdown()
    {
        assert( _HostBuffer );
        CUDA_REQUIRE(( cudaFreeHost( _HostBuffer ) ));
        _HostBuffer = NULL;

        assert( _DeviceBuffer );
        CUDA_REQUIRE(( cudaFree( _DeviceBuffer ) ));        
        _DeviceBuffer = NULL;
    }

    template< typename T >
    void Alloc( size_t count, CudaBuffer< T >* dest )
    {
        size_t size = count * sizeof( T );
        size_t offset = _Heap.Alloc( size );

        dest->_Host   = (T*) ((uintptr_t) _HostBuffer   + offset);
        dest->_Device = (T*) ((uintptr_t) _DeviceBuffer + offset);
        dest->_Offset = offset;
        dest->_BufferSize = size;                                 
    }

    template< typename T >
    void Free( CudaBuffer< T >& buf )
    {
        if( !buf.IsNull() )
        {
            _Heap.Free( buf._Offset );
            buf.SetNull();
        }
    }
};

static inline int GetCudaCoresPerSM( int major, int minor )
{
    switch( (major << 4) + minor )
    {
    case 0x30:
    case 0x32:
    case 0x35:
    case 0x37: 
        return 192;

    case 0x50:
    case 0x52:
    case 0x53:
    case 0x61:
    case 0x62: 
        return 128;

    case 0x60:
    case 0x70:
    case 0x72:
    case 0x75: 
        return  64;
    }

    return 64; // ?? FIXME
}