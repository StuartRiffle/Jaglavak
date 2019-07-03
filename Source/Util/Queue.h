// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T >
class ThreadSafeQueue
{
    mutex mMutex;
    condition_variable mVar;
    list< T > mQueue;
    volatile bool mShuttingDown;

public:
    ThreadSafeQueue()
    {
        mShuttingDown = false;
    }

    ~ThreadSafeQueue()
    {
        this->Terminate();
    }

    void Terminate()
    {
        mShuttingDown = true;
        mVar.notify_all();
    }

    void Push( const T* objs, size_t count )
    {
        unique_lock< mutex > lock( mMutex );

        mQueue.insert( mQueue.end(), objs, objs + count );

        lock.unlock();
        mVar.notify_all();
    }

    void Push( const vector< T >& elems )
    {
        this->Push( elems.data(), elems.size() );
    }

    void Push( const T& obj )
    {
        this->Push( &obj, 1 );
    }

    bool Pop( T& result, bool blocking = true )
    {
        unique_lock< mutex > lock( mMutex );

        if( blocking )
        {
            while( mQueue.empty() )
            {
                mVar.wait( lock );

                if( mShuttingDown )
                    return false;
            }
        }

        if( mQueue.empty() )
            return false;

        result = mQueue.front();
        mQueue.pop_front();
        return true;
    }

    bool TryPop( T& result )
    {
        bool blocking = false;
        return( this->Pop( result, blocking ) );
    }

    vector< T > PopMulti( size_t limit )
    {
        unique_lock< mutex > lock( mMutex );

        vector< T > result;
        result.reserve( limit );

        while( mQueue.size() < limit )
        {
            mVar.wait( lock );

            if( mShuttingDown )
                return result;
        }

        while( result.size() < limit ) 
        {
            assert( !mQueue.empty() );
            result.push_back( mQueue.front() );
            mQueue.pop_front();
        }

        return result;
    }

    vector< T > PopAll()
    {
        unique_lock< mutex > lock( mMutex );

        vector< T > result( mQueue.begin(), mQueue.end() );
        mQueue.clear();

        return result;
    }

    size_t PeekCount()
    {
        unique_lock< mutex > lock( mMutex );
        return mQueue.size();
    }
};

