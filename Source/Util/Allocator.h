// JAGLAVAK CHESS ENGINE (c) 2019 Stuaraddr_t Riffle
#pragma once

typedef uintptr_t addr_t;

class HeapAllocator
{
    typedef map< addr_t, size_t > AddrToSizeMap;

    AddrToSizeMap   _Free;
    AddrToSizeMap   _Used;
    mutex           _Mutex;
    size_t          _Size;
    size_t          _Base;
    size_t          _Align;
    size_t          _TotalAllocated;
    size_t          _HighestAllocated;

    bool IsAligned( addr_t addr ) const { return( (addr & (_Align - 1)) == 0 ); }

public:
    const addr_t INVALID = addr_t( ~0 );

    void Init( size_t size, addr_t base = 0, size_t alignment = 128 )
    {
        assert( (alignment & (alignment - 1)) == 0 );
        _Align = alignment;

        assert( IsAligned( base ) );
        assert( IsAligned( size ) );

        if( base == 0 )
        {
            // 0 is reserved to mean NULL

            base += _Align;
            size -= _Align;
        }

        _Align = alignment;
        _Size = size;
        _Base = base;
        _TotalAllocated = 0;
        _HighestAllocated = 0;

        _Used.clear();
        _Free.clear();
        _Free[base] = size;

        this->SanityCheck();
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
            size_t& freeSize = iter->second;

            auto next = iter;
            if( ++next != _Free.end() )
            {
                addr_t nextAddr = next->first;
                size_t nextSize = next->second;

                if( nextAddr == (freeAddr + freeSize) )
                {
                    freeSize += nextSize;
                    _Free.erase( next );
                    continue;
                }
            }

            if( freeSize >= size )
            {
                addr_t addr = freeAddr;

                assert( _Used.find( addr ) == _Used.end() );
                if( size > 0 )
                    _Used[addr] = size;

                if( size < freeSize )
                    _Free[freeAddr + size] = freeSize - size;
                _Free.erase( iter );

                _TotalAllocated += size;
                if( _TotalAllocated > _HighestAllocated )
                    _HighestAllocated = _TotalAllocated;

                this->SanityCheck();
                return addr;
            }

            iter = next;
        }

        assert( !"Heap overflow" );
        return 0;
    }

    void Free( addr_t addr )
    {
        unique_lock< mutex > lock( _Mutex );

        assert( addr != 0 );

        assert( addr >= _Base );

        assert( addr < _Base + _Size );

        addr_t mask = (_Align - 1); 
        addr_t result = addr & mask;
        assert( result == 0 );

        auto iter = _Used.find( addr );
        assert( iter != _Used.end() );
        assert( _Free.find( addr ) == _Free.end() );

        size_t size = iter->second;
        assert( _TotalAllocated >= size );
        _TotalAllocated -= size;

        _Free[addr] = size;
        _Used.erase( iter );

        this->SanityCheck();
    }

    void SanityCheck()
    {
#if DEBUG
        size_t checkAllocated = 0;
        for( auto iter : _Used )
            checkAllocated += iter.second;

        assert( _TotalAllocated == checkAllocated );
        assert( _TotalAllocated <= _HighestAllocated );
        assert( _HighestAllocated <= _Size );

        // Make sure that between them, _Free and _Used account for the
        // entire address space, with no gaps or overlap

        vector< pair< addr_t, size_t > > blocks;
        blocks.reserve( _Free.size() + _Used.size() );

        for( auto iter : _Free )
            blocks.push_back( iter );

        for( auto iter : _Used )
            blocks.push_back( iter );

        std::sort( blocks.begin(), blocks.end() );

        addr_t cursor = _Base;
        for( auto& block : blocks )
        {
            addr_t addr = block.first;
            size_t size = block.second;

            assert( (addr & (_Align - 1)) == 0 );
            assert( addr == cursor );
            assert( size >= _Align );

            cursor += size;
        }

        assert( cursor == (_Base + _Size) );
#endif
    }
};

