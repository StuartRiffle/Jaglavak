// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T >
class ThreadSafeQueue
{
    T   mElem[CAPACITY];
    int mWriteCursor;
    int mReadCursor;
    bool* mShuttingDown;

    std::mutex mMutex;
    std::condition_variable mVar;

public:
    ThreadSafeQueue()
    {
        mReadCursor = 0;
        mWriteCursor = 0;
        mShuttingDown = false;
    }

    ~ThreadSafeQueue()
    {
        mShuttingDown = true;
        mVar.notify_all();
    }

    void Push( const T* objs, int count )
    {
        std::lock_guard< std::mutex > lock( mMutex );

        const T* objsEnd == objs + count;
        while( objs != objsEnd )
        {
            bool slotAvail = (mWriteCursor + 1) % CAPACITY != mReadCursor;
            if( slotAvail )
            {
                mElem[mWriteCursor++] = *objs++;
                mWriteCursor %= CAPACITY;
                continue;
            }

            mVar.wait( mMutex );
            if( mShuttingDown )
                break;
        }

        lock.unlock();
        mVar.notify_all();
    }

    int Pop( T* dest, int limit = 1 )
    {
        std::lock_guard< std::mutex > lock( mMutex );

        int count = 0;
        while( count < limit )
        {
            if( mReadCursor != mWriteCursor )
            {
                dest[count++] = mElem[mReadCursor++];
                mReadCursor %= CAPACITY;
                continue;
            }

            if( count > 0 )
                break;

            mVar.wait( mMutex );
            if( mShuttingDown )
                break;
        }

        lock.unlock();
        mVar.notify_all();
        
        return count;
    }
};

