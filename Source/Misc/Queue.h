// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T >
class ThreadSafeQueue
{
    std::vector< T > mBuffer;
    size_t mCount;
    size_t mWrapMask;
    size_t mWriteCursor;

    std::mutex mMutex;
    std::condition_variable mVar;
    volatile bool* mShuttingDown;

public:
    ThreadSafeQueue( size_t capacity )
    {
        assert( capacity & (capacity - 1) == 0 );

        mBuffer.resize( capacity );
        mCount = 0;
        mWrapMask = capacity - 1;
        mWriteCursor = 0;
        mShuttingDown = false;
    }

    ~ThreadSafeQueue()
    {
        mShuttingDown = true;
        mVar.notify_all();    
    }

    size_t PushBatch( const T* objs, size_t count, size_t minimum )
    {
        std::lock_guard< std::mutex > lock( mMutex );

        size_t numPushed = 0;
        while( numPushed < count )
        {
            size_t capacity = mBuffer.size();
            if( mCount < capacity )
            {
                mBuffer[mWriteCursor++ & mWrapMask] = objs[numPushed++];
                mCount++;
                continue;
            }

            if( numPushed >= minimum )
                break;

            mVar.wait( mMutex );
            if( mShuttingDown )
                break;
        }

        lock.unlock();
        mVar.notify_all();

        return numPushed;
    }

    void PushBatch( const T* objs, size_t count )
    {
        this->PushBatch( objs, count, count );
    }

    void PushBatch( const std::vector< T >& elems )
    {
        this->PushBatch( elems.data(), elems.size() );
    }

    void Push( const T& obj )
    {
        this->PushBatch( &obj, 1 );
    }

    size_t PopBatch( T* dest, size_t limit, size_t minimum )
    {
        std::lock_guard< std::mutex > lock( mMutex );

        size_t numPopped = 0;
        size_t readCursor = mWriteCursor - mCount;

        while( numPopped < limit )
        {
            if( mCount > 0 )
            {
                dest[numPopped++] = mBuffer[readCursor++ & mWrapMask];
                mCount--;
                continue;
            }

            if( numPopped >= minimum )
                break;

            mVar.wait( mMutex );
            if( mShuttingDown )
                break;
        }

        lock.unlock();
        mVar.notify_all();

        return numPopped;
    }

    std::vector< T > PopBatch( size_t count, size_t minimum = 1 )
    {
        std::vector< T > result;
        result.resize( count );

        size_t numPopped = this->PopBatch( result.data(), limit, minimum );
        result.resize( numPopped );

        return result;
    }

    T Pop()
    {
        T result;
        this->PopBatch( &result, 1 );
        return result;
    }
};

