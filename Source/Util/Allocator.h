// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T = uintptr_t >
class HeapAllocator
{
    typedef map< T, T > AddrToSizeMap;

    AddrToSizeMap   _Free;
    AddrToSizeMap   _Used;
    mutex           _Mutex;
    T               _Range;
    T               _Align;
    T               _TotalAllocated;
    T               _HighestAllocated;

public:
    const T INVALID = T( ~0 );

    void Init( T range, T base = 0, T alignment = 128 )
    {
        assert( (alignment & (alignment - 1)) == 0 );
        assert( (base      & (alignment - 1)) == 0 );
        assert( (range     & (alignment - 1)) == 0 );

        _Range = range;
        _Align = alignment;
        _TotalAllocated = 0;
        _HighestAllocated = 0;

        _Used.clear();
        _Free.clear();
        _Free[base] = range;
    }

    T Alloc( T size )
    {
        unique_lock< mutex > lock( _Mutex );

        size = (size + _Align - 1) & ~(_Align - 1);

        auto iter = _Free.begin();
        while( iter != _Free.end() )
        {
            T freeAddr = iter->first;
            T freeSize = iter->second;

            auto next = iter;
            if( ++next != _Free.end() )
            {
                T nextAddr = next->first;
                T nextSize = next->second;

                if( nextAddr == (freeAddr + freeSize) )
                {
                    iter->second += nextSize;
                    _Free.erase( next );
                    continue;
                }
            }

            if( size <= freeSize )
            {
                T addr = freeAddr;

                assert( _Used.find( addr ) == _Used.end() );
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

        assert( !"Out of CUDA heap" );
        return INVALID;
    }

    void Free( T addr )
    {
        unique_lock< mutex > lock( _Mutex );

        auto iter = _Used.find( addr );
        assert( iter != _Used.end() );
        assert( _Free.find( addr ) == _Free.end() );

        T size = iter->second;
        assert( _TotalAllocated >= size );
        _TotalAllocated -= size;

        _Free[addr] = size;
        _Used.erase( iter );
    }

    void DebugValidate()
    {
#if DEBUG
        auto iterUsed = _Used.begin();
        auto iterFree = _Free.begin();

        size_t offset = 0;

        for( ;; )
        {
            if( iterFree != _Free.end() )
            {
                if(iterFree->first == offset)
                {
                    offset += iterFree->second;
                    ++iterFree;
                    continue;
                }
            }

            if( iterUsed != _Used.end() )
            {
                if(iterUsed->first == offset)
                {
                    offset += iterUsed->second;
                    ++iterUsed;
                    continue;
                }
            }

            if( iterFree == _Free.end() && iterUsed == _Used.end() )
                break;

            assert( 0 );
        }

        assert( offset == _Range );
#endif
    }
};

