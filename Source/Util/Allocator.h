// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T = uintptr_t >
class HeapAllocator
{
    typedef map< T, T > AddrToSizeMap;

    AddrToSizeMap   mFree;
    AddrToSizeMap   mUsed;
    mutex           mMutex;
    T               mRange;
    T               mAlign;
    T               mTotalAllocated;
    T               mHighestAllocated;

public:
    const T INVALID = T( ~0 );

    void Init( T range, T base = 0, T alignment = 128 )
    {
        assert( (alignment & (alignment - 1)) == 0 );
        assert( (base      & (alignment - 1)) == 0 );
        assert( (range     & (alignment - 1)) == 0 );

        mRange = range;
        mAlign = alignment;
        mTotalAllocated = 0;
        mHighestAllocated = 0;

        mUsed.clear();
        mFree.clear();
        mFree[base] = range;
    }

    T Alloc( T size )
    {
        unique_lock< mutex > lock( mMutex );

        size = (size + mAlign - 1) & ~(mAlign - 1);

        auto iter = mFree.begin();
        while( iter != mFree.end() )
        {
            T freeAddr = iter->first;
            T freeSize = iter->second;

            auto next = iter;
            if( ++next != mFree.end() )
            {
                T nextAddr = next->first;
                T nextSize = next->second;

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

                mFree.erase( iter );
                if( size < freeSize )
                    mFree[freeAddr + size] = freeSize - size;

                mTotalAllocated += size;
                if( mTotalAllocated > mHighestAllocated )
                    mHighestAllocated = mTotalAllocated;

                return addr;
            }

            iter = next;
        }

        assert( !"Out of CUDA heap" );
        return INVALID;
    }

    void Free( T addr )
    {
        unique_lock< mutex > lock( mMutex );

        auto iter = mUsed.find( addr );
        assert( iter != mUsed.end() );
        assert( mFree.find( addr ) == mFree.end() );

        T size = iter->second;
        assert( mTotalAllocated >= size );
        mTotalAllocated -= size;

        mFree[addr] = size;
        mUsed.erase( iter );
    }

    void DebugValidate()
    {
#if DEBUG
        auto iterUsed = mUsed.begin();
        auto iterFree = mFree.begin();

        size_t offset = 0;

        for( ;; )
        {
            if( iterFree != mFree.end() )
            {
                if(iterFree->first == offset)
                {
                    offset += iterFree->second;
                    ++iterFree;
                    continue;
                }
            }

            if( iterUsed != mUsed.end() )
            {
                if(iterUsed->first == offset)
                {
                    offset += iterUsed->second;
                    ++iterUsed;
                    continue;
                }
            }

            if( iterFree == mFree.end() && iterUsed == mUsed.end() )
                break;

            assert( 0 );
        }

        assert( offset == mRange );
#endif
    }
};

