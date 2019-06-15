// JAGLAVAK CHESS ENGINE (c) 2019 Stuaraddr_t Riffle
#pragma once

typedef uintptr_t addr_t;
#define INVALID_ADDR (addr_t( ~0 ))

class HeapAllocator
{
    typedef map< addr_t, size_t > AddrToSizeMap;

    AddrToSizeMap   _Free;
    AddrToSizeMap   _Used;
    mutex           _Mutex;
    size_t          _Align;
    size_t          _TotalAllocated;
    size_t          _HighestAllocated;

public:
    const addr_t INVALID = addr_t( ~0 );

    void Init( size_t size, addr_t base = 0, size_t alignment = 128 )
    {
        assert( (alignment & (alignment - 1)) == 0 );
        assert( (base      & (alignment - 1)) == 0 );
        assert( (size      & (alignment - 1)) == 0 );

        _Align = alignment;
        _TotalAllocated = 0;
        _HighestAllocated = 0;

        _Used.clear();
        _Free.clear();
        _Free[base] = size;
    }

    addr_t Alloc( size_t size )
    {
        unique_lock< mutex > lock( _Mutex );

        size = (size + _Align - 1) & ~(_Align - 1);
        assert( size > 0 );

        auto iter = _Free.begin();
        while( iter != _Free.end() )
        {
            addr_t freeAddr = iter->first;
            size_t freeSize = iter->second;

            auto next = iter;
            if( ++next != _Free.end() )
            {
                addr_t nextAddr = next->first;
                size_t nextSize = next->second;

                if( nextAddr == (freeAddr + freeSize) )
                {
                    // Combine consecutive free blocks

                    iter->second += nextSize;
                    _Free.erase( next );
                    continue;
                }
            }

            if( size <= freeSize )
            {
                addr_t addr = freeAddr;

                assert( _Used.find( addr ) == _Used.end() );
                if( size > 0 )
                    _Used[addr] = size;

                _Free.erase( iter );
                if( size < freeSize )
                    _Free[freeAddr + size] = freeSize - size;

                _TotalAllocated += size;
                if( _TotalAllocated > _HighestAllocated )
                    _HighestAllocated = _TotalAllocated;

                return addr;
            }

            iter = next;
        }

        assert( !"Heap overflow" );
        return INVALID_ADDR;
    }

    void Free( addr_t addr )
    {
        unique_lock< mutex > lock( _Mutex );

        auto iter = _Used.find( addr );
        assert( iter != _Used.end() );
        assert( _Free.find( addr ) == _Free.end() );

        size_t size = iter->second;
        assert( _TotalAllocated >= size );
        _TotalAllocated -= size;

        _Free[addr] = size;
        _Used.erase( iter );
    }
};

