// WIP
#pragma once


struct LargePage
{
    void* _Ptr;
    size_t _Size;

    LargePageAlloc( size_t size ) : _Size( size )
    {
#if TOOLCHAIN_GCC
        _Ptr = mmap( NULL, _Size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0 );
#elif TOOLCHAIN_MSVC
        _Ptr = VirtualAlloc( NULL, _Size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE );
#else
        _Ptr = memalign( _Size, _Size );
#endif
    }

    ~HugePage()
    {
#if TOOLCHAIN_GCC
        munmap( _Ptr, _Size );
#elif TOOLCHAIN_MSVC
        VirtualFree( _Ptr );
#else
        free( _Ptr );
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

