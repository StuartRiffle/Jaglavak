// Threads.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
        
#pragma once

struct Semaphore
{
#if TOOLCHAIN_MSVC
    HANDLE      mHandle;
    Semaphore() : mHandle( CreateSemaphore( NULL, 0, LONG_MAX, NULL ) ) {}
    ~Semaphore() { CloseHandle( mHandle ); }
    void Post() { ReleaseSemaphore( mHandle, 1, NULL ); }
    void Wait() { WaitForSingleObject( mHandle, INFINITE ); }
#elif TOOLCHAIN_GCC
    sem_t       mHandle;
    Semaphore()  { sem_init( &mHandle, 0, 0 ); }
    ~Semaphore() { sem_destroy( &mHandle ); }
    void Post( int count = 1 )  { while( count-- ) sem_post( &mHandle ); }
    void Wait() { while( sem_wait( &mHandle ) ); }
#endif
};

struct Mutex
{
#if TOOLCHAIN_MSVC
    CRITICAL_SECTION mCritSec;
    Mutex()      { InitializeCriticalSection( &mCritSec ); }
    ~Mutex()     { DeleteCriticalSection( &mCritSec ); }
    void Enter() { EnterCriticalSection( &mCritSec ); }
    void Leave() { LeaveCriticalSection( &mCritSec ); }
#elif TOOLCHAIN_GCC
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

#define MUTEX_SCOPE( _VAR ) Mutex::Scope lock( _VAR );

template< typename T >
class ThreadSafeQueue
{
    Mutex           mMutex;
    Semaphore       mAvail;
    std::list< T >  mQueue;

public:
    int GetCount()
    {
        MUTEX_SCOPE( mMutex );

        return (int) mQueue.size();
    }

    void Push( const T& obj )
    {
        mMutex.Enter();
        mQueue.push_back( obj );
        mMutex.Leave();

        mAvail.Post();
    }

    T Pop()
    {
        static ThreadSafeQueue* thishere;
        thishere = this;

        mAvail.Wait();

        MUTEX_SCOPE( mMutex );

        assert( !mQueue.empty() );
        T result = mQueue.front();
        mQueue.pop_front();

        return( result );
    }
};

