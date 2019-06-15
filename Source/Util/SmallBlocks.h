// WIP
#pragma once


struct LargePage
{
    addr_t _Addr;
    size_t _Size;
    size_t _Used;

    HugePage() : _Addr( 0 ), _Size( 0 ), _Used( 0 )
    {
#if TOOLCHAIN_GCC
        _Size = (size_t) sysconf( _SC_PAGESIZE );
        _Addr = (addr_t) mmap( NULL, _Size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0 );
#elif TOOLCHAIN_MSVC
        _Size = GetLargePageMinimum();
        _Addr = (addr_t) VirtualAlloc( NULL, _Size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE );
#else
        _Size = DEFAULT_SIZE;
        _Addr = (addr_t) memalign( _Size, _Size );
#endif
    }

    ~HugePage()
    {
#if TOOLCHAIN_GCC
        munmap( (void*) _Addr, _Size );
#elif TOOLCHAIN_MSVC
        VirtualFree( (void*) _Addr );
#else
        free( (void*) _Addr );
#endif
    }
};

class SmallBlockAllocator
{
    list< HugePage >  _Pages;
    vector< addr_t* > _FreeBlocks;

    SmallBlockAllocator()
    {
        _Pages.push_back();
    }

    void* Alloc( size_t bytes )
    {
        size_t slot = ROUND_UP_POW2( bytes, _Alignment ) >> _AlignShift;
        if( slot >= _FreeBlocks.size() )
            _FreeBlocks.resize( slot + 1 );

        void** ptr = _FreeBlocks[slot];
        if( ptr )
        {
            addr_t* next = *ptr;
            _FreeBlocks[slot] = next;
            return ptr;
        }

        if( mCursor + bytes > mPages.front()._Size )
            _Pages.push_front();

        HugePage& page = _Pages.front();
        {
            ptr = (void*) (page._Addr + _Offset);
            _Offset += bytes;
            return ptr;
        }

        _Pages.push_front();


        // Cut off a slice

    }
        else
        {
        void* ptr = this->CutSlice( bytes );
        mLargeBlocks[bytes].
        }



        int slot 
}

void Free(void* ptr)
{

}

};

