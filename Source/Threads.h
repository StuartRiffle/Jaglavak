// Threads.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle
        
#ifndef CORVID_THREAD_H__
#define CORVID_THREAD_H__

struct Semaphore
{
#if CORVID_MSVC
    HANDLE      mHandle;
    Semaphore() : mHandle( CreateSemaphore( NULL, 0, LONG_MAX, NULL ) ) {}
    ~Semaphore() { CloseHandle( mHandle ); }
    void Post( int count ) { ReleaseSemaphore( mHandle, count, NULL ); }
    void Wait( int timeout = INFINITE ) { return( WaitForSingleObject( mHandle, timeout ) != WAIT_TIMEOUT); }
#elif CORVID_GCC
    sem_t       mHandle;
    Semaphore()  { sem_init( &mHandle, 0, 0 ); }
    ~Semaphore() { sem_destroy( &mHandle ); }
    void Post( int count )  { while( count-- ) sem_post( &mHandle ); }

    bool Wait( int timeout = -1 )
    { 
        if( timeout < 0 )
        {
            while( sem_wait( &mHandle ) ) {}
            return;
        }

        timespec tv = { 0, timeout * 1000000 };
        return (sem_timedwait( &mHandle, &tv ) == 0);
    }
#endif
};

struct Mutex
{
#if CORVID_MSVC
    CRITICAL_SECTION mCritSec;
    Mutex()      { InitializeCriticalSection( &mCritSec ); }
    ~Mutex()     { DeleteCriticalSection( &mCritSec ); }
    void Enter() { EnterCriticalSection( &mCritSec ); }
    void Leave() { LeaveCriticalSection( &mCritSec ); }
#elif CORVID_GCC
    pthread_mutexattr_t mAttrib;
    pthread_mutex_t     mMutex;
    Mutex()      { pthread_mutexattr_init( &mAttrib ); pthread_mutexattr_settype( &mAttrib, PTHREAD_MUTEX_RECURSIVE ); pthread_mutex_init( &mMutex, &mAttrib ); }
    ~Mutex()     { pthread_mutex_destroy( &mMutex ); pthread_mutexattr_destroy( &mAttrib ); }
    void Enter() { pthread_mutex_lock( &mMutex ); }
    void Leave() { pthread_mutex_unlock( &mMutex ); }
#endif

    class Scope
    {
        Mutex& mMutex;
    public:
        Scope( Mutex& mutex ) : mMutex( mutex ) { mMutex.Enter(); }
        ~Scope() { mMutex.Leave(); }
    };
};


template< typename T >
class ThreadSafeQueue
{
    Mutex           mMutex;
    Semaphore       mAvail;
    std::list< T >  mQueue;

public:
    void Push( const T& obj )
    {
        this->Push( &obj, 1 );
    }   

    void Push( const T* objs, size_t count )
    {
        Mutex::Scope lock( mMutex );

        mQueue.append( objs, objs + count );

        mAvail.Post(count);
    }


    T Pop()
    {
        mAvail.Wait();

        Mutex::Scope lock( mMutex );

        T result = mQueue.front();
        mQueue.pop_front();

        return( result );
    }

    bool TryPop( T& result )
    {
        Mutex::Scope lock( mMutex );

        if( mQueue.empty() )
            return( false );

        mAvail.Wait();

        result = mQueue.front();
        mQueue.pop_front();

        return( true );
    }

    vector< T > PopAll()
    {
        Mutex::Scope lock( mMutex );
        vector< T > result;

        if(!mQueue.empty())
        {
            size_t count = mQueue.size();

            result.reserve(count);
            result.append(mQueue.begin(), mQueue.end());

            mQueue.clear();
            mAvail.Wait(count);
        }

        return( result );
    }

    void Clear()
    {
        Mutex::Scope lock( mMutex );

        size_t count = mQueue.size();

        mQueue.clear();
        mAvail.Wait(count);
    }
};


#endif // CORVID_THREAD_H__
