// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T, int CAPACITY = 8192 >
class ThreadSafeQueue
{
    T               mElem[CAPACITY];
    int             mReadCursor;
    int             mWriteCursor;
    volatile bool*  mShuttingDown;
    std::mutex      mMutex;
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

    void Push( const T& obj )
    {
        this->Push( &obj, 1 );
    }

    int Pop( T* dest, int limit, bool blocking = true )
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

            if( !blocking )
                break;

            mVar.wait( mMutex );

            if( mShuttingDown )
                break;
        }

        lock.unlock();
        mVar.notify_all();

        return count;
    }

    T Pop()
    {
        T result;

        this->Pop( &result, 1 );
        return result;
    }

    std::vector PopBulk( int limit = 1024 )
    {
        std::vector< T > result;
        result.resize( limit );

        int count = this->Pop( result.data(), limit, false );
        result.resize( count );

        return result;
    }
};

