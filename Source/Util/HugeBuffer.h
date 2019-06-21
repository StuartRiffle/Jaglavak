// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct HugeBuffer
{
    void* _Ptr;
    size_t _Size;

    HugeBuffer( size_t size, bool useLargePages = true ) : _Size( size )
    {
#if TOOLCHAIN_GCC
        // FIXME: handle huge page failure etc
        _Ptr = mmap( NULL, _Size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0 );
#elif TOOLCHAIN_MSVC
        DWORD flags = MEM_RESERVE | MEM_COMMIT;
        size_t pageSize = GetLargePageMinimum();

        if( (pageSize > 0) && useLargePages )
        {
            size = ((size + pageSize - 1) / pageSize) * pageSize;
            _Ptr = VirtualAlloc( NULL, size, flags | MEM_LARGE_PAGES, PAGE_READWRITE );
            if( _Ptr )
                return;
        }

        _Ptr = VirtualAlloc( NULL, size, flags, PAGE_READWRITE );
        return;
#endif
    }

    ~HugeBuffer()
    {
#if TOOLCHAIN_GCC
        munmap( _Ptr, _Size );
#elif TOOLCHAIN_MSVC
        VirtualFree( _Ptr, 0, MEM_RELEASE );
#endif
    }
};

