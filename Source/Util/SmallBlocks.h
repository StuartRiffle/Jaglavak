// WIP
#pragma once


struct HugePage
{
    addr_t _Addr;
    size_t _Size;

    HugePage()
    {
        #if TOOLCHAIN_GCC
        _Size = (size_t) sysconf( _SC_PAGESIZE );
        _Addr = (addr_t) mmap( NULL, _Size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0 );
        #elif TOOLCHAIN_MSVC
        _Size = GetLargePageMinimum();
        _Addr = (addr_t) VirtualAlloc( NULL, _Size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE );
        #endif
    }

    ~HugePage()
    {
        #if TOOLCHAIN_GCC
        munmap( (void*) _Addr, _Size );
        #elif TOOLCHAIN_MSVC
        VirtualFree( (void*) _Addr );
        #endif
    }
};

class SlabAllocator
{
    list< HugePage > mSlabs;
    vector< addr_t* > _FreeBlocks;



    uintptr_t mSlabSize;
    uintptr_t mCursor;

    SlabAllocator()
    {
        _Slabs.push_back();
    }

    void* Alloc( size_t bytes )
    {
        size_t slot = ROUND_UP_POW2( bytes, _Alignment ) >> _AlignShift;
        if( slot >= _FreeBlocks.size() )
            _FreeBlocks.resize[slot];

        void* ptr = _FreeBlocks[slot];
        if( ptr )
        {
            addr_t* next = *ptr;
            _FreeBlocks[slot] = next;
            return ptr;
        }

        if( mCursor + bytes > mSlabs.front()._Size )
            _Slabs.push_front();

        HugePage& page = _Slabs.front();

        {
            ptr = (void*) (page._Addr + _Offset);
            _Offset += bytes;
            return ptr;
        }

        _Slabs.push_front();


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

