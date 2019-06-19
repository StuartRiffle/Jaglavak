// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct HugePageAlloc
{
    void* _Ptr;
    size_t _Size;

    HugePageAlloc( size_t size ) : _Size( size )
    {
#if TOOLCHAIN_GCC
        _Ptr = mmap( NULL, _Size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0 );
#elif TOOLCHAIN_MSVC
        _Ptr = VirtualAlloc( NULL, _Size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE );
#else
        _Ptr = memalign( _Size, 4096 );
#endif
    }

    ~HugePageAlloc()
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
