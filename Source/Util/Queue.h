// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T >
class ThreadSafeQueue
{
    list< T >           _Queue;
    mutex               _Mutex;
    condition_variable  _Var;
    volatile bool       _ShuttingDown;

public:
    ThreadSafeQueue()
    {
        _ShuttingDown = false;
    }

    ~ThreadSafeQueue()
    {
        this->Terminate();
    }

    void Terminate()
    {
        _ShuttingDown = true;
        NotifyAllWaiters();
    }

    void NotifyAllWaiters()
    {
        _Var.notify_all();
    }

    void Push( const T* objs, size_t count )
    {
        unique_lock< mutex > lock( _Mutex );

        _Queue.insert( _Queue.end(), objs, objs + count );

        lock.unlock();
        _Var.notify_all();
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
        unique_lock< mutex > lock( _Mutex );

        if( blocking )
        {
            while( _Queue.empty() )
            {
                _Var.wait( lock );
                if( _ShuttingDown )
                    return false;
            }
        }

        if( _Queue.empty() )
            return false;

        result = _Queue.front();
        _Queue.pop_front();
        return true;
    }

    bool TryPop( T& result )
    {
        bool blocking = false;
        return( this->Pop( result, blocking ) );
    }

    vector< T > PopMulti( size_t limit )
    {
        unique_lock< mutex > lock( _Mutex );

        vector< T > result;
        result.reserve( limit );

        while( _Queue.size() < limit )
        {
            _Var.wait( lock );
            if( _ShuttingDown )
                return result;
        }

        while( (result.size() < limit) && !_Queue.empty() )
        {
            result.push_back( _Queue.front() );
            _Queue.pop_front();
        }

        return result;
    }

    vector< T > PopAll()
    {
        unique_lock< mutex > lock( _Mutex );

        vector< T > result( _Queue.begin(), _Queue.end() );
        _Queue.clear();

        return result;
    }

    size_t PeekCount()
    {
        unique_lock< mutex > lock( _Mutex );
        return _Queue.size();
    }
};

