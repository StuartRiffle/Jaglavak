// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T >
class ThreadSafeQueue
{
    enum
    {
        DEFAULT_BATCH_SIZE = 1024,
        DEFAULT_CAPACITY = 8192,
    };

    vector< T > mBuffer;
    size_t mCount;
    size_t mWrapMask;
    size_t mWriteCursor;

    mutex mMutex;
    condition_variable mVar;
    volatile bool mShuttingDown;

public:
    ThreadSafeQueue( size_t capacity = DEFAULT_CAPACITY )
    {
        assert( (capacity & (capacity - 1)) == 0 );

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

    size_t PushMulti( const T* objs, size_t count, size_t minimum )
    {
        unique_lock< mutex > lock( mMutex );

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

            mVar.wait( lock );
            if( mShuttingDown )
                break;
        }

        lock.unlock();
        mVar.notify_all();

        return numPushed;
    }

    void PushMulti( const T* objs, size_t count )
    {
        this->PushMulti( objs, count, count );
    }

    void PushMulti( const vector< T >& elems )
    {
        this->PushMulti( elems.data(), elems.size() );
    }

    void Push( const T& obj )
    {
        this->PushMulti( &obj, 1 );
    }

    size_t PopMulti( T* dest, size_t limit, size_t minimum )
    {
        unique_lock< mutex > lock( mMutex );

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

            mVar.wait( lock );
            if( mShuttingDown )
                break;
        }

        lock.unlock();
        mVar.notify_all();

        return numPopped;
    }

    vector< T > PopMulti( size_t limit = DEFAULT_BATCH_SIZE )
    {
        vector< T > result;
        result.resize( limit );

        size_t numPopped = this->PopMulti( result.data(), limit, 0 );
        result.resize( numPopped );

        return result;
    }

    bool PopBlocking( T& result )
    {
        size_t success = this->PopMulti( &result, 1, 1 );
        return success;
    }
};

