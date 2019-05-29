// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T = uintptr_t >
class HeapAllocator
{
    typedef map< T, T > AddrToSizeMap;

    AddrToSizeMap   mFree;
    AddrToSizeMap   mUsed;
    mutex           mMutex;
    T               mAlign;

public:
    const T INVALID = T( ~0 );

    void Init( T range, T base = 0, T align = 128 )
    {
        assert( align & (align - 1) == 0 );
        assert( base  & (align - 1) == 0 );
        assert( range & (align - 1) == 0 );

        mAlign = align;
        mUsed.clear();
        mFree.clear();
        mFree[base] = range;
    }

    T Alloc( T size )
    {
        unique_lock< mutex > lock( mMutex );

        size = (size + mAlign - 1) & (mAlign - 1);

        auto iter = mFree.begin();
        while( iter != mFree.end() )
        {
            T freeAddr = iter->first;
            T freeSize = iter->second;

            auto next = iter;
            ++next;

            if( next != mFree.end() )
            {
                T nextAddr = next->first;
                T nextSize = next->second;

                // Combine adjacent free blocks

                if( nextAddr == (freeAddr + freeSize) )
                {
                    iter->second += nextSize;
                    mFree.erase( next );
                    continue;
                }
            }

            if( size <= freeSize )
            {
                T addr = freeAddr;
                assert( mUsed.find( addr ) == mUsed.end() );
                mUsed[addr] = size;

                if( size < freeSize )
                    mFree[freeAddr + size] = freeSize - size;

                mFree.erase( iter );
                return addr;
            }

            iter = next;
        }

        assert( 0 );
        return INVALID;
    }

    void Free( T addr )
    {
        unique_lock< mutex > lock( mMutex );

        auto iter = mUsed.find( addr );
        assert( iter != mUsed.end() );
        assert( mFree.find( addr ) == mFree.end() );

        T size = iter->second;
        mFree[addr] = size;
        mUsed.erase( iter );
    }
};

