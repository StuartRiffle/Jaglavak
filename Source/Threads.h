// Threads.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle
        
#ifndef CORVID_THREAD_H__
#define CORVID_THREAD_H__

struct Semaphore
{
#if TOOLCHAIN_MSVC
    HANDLE      mHandle;
    Semaphore() : mHandle( CreateSemaphore( NULL, 0, LONG_MAX, NULL ) ) {}
    ~Semaphore() { CloseHandle( mHandle ); }
    void Post( int count = 1 ) { ReleaseSemaphore( mHandle, count, NULL ); }
    void Wait() { WaitForSingleObject( mHandle, INFINITE ); }
#elif TOOLCHAIN_GCC
    sem_t       mHandle;
    Semaphore()  { sem_init( &mHandle, 0, 0 ); }
    ~Semaphore() { sem_destroy( &mHandle ); }
    void Post( int count = 1 )  { while( count-- ) sem_post( &mHandle ); }
    bool Wait() { while( sem_wait( &mHandle ) ); }
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
        this->Push( &obj, 1 );
    }   

    void Push( const T* objs, size_t count )
    {
        MUTEX_SCOPE( mMutex );

        mQueue.insert( mQueue.end(), objs, objs + count );
        mAvail.Post( (int) count );
    }

    T Pop()
    {
        // Blocking

        mAvail.Wait();
        {
            MUTEX_SCOPE( mMutex );

            T result = mQueue.front();
            mQueue.pop_front();

            return( result );
        }
    }

    bool TryPop( T& result )
    {
        MUTEX_SCOPE( mMutex );

        if( mQueue.empty() )
            return( false );

        mAvail.Wait();

        result = mQueue.front();
        mQueue.pop_front();

        return( true );
    }

    std::vector< T > PopMultiple( size_t limit )
    {
        // Blocking

        mAvail.Wait();
        {
            MUTEX_SCOPE( mMutex );

            size_t count = Min( mQueue.size(), limit );

            std::vector< T > result;
            result.reserve( count );

            for( size_t i = 0; i < count; i++ )
            {
                result.push_back( mQueue.front() );
                mQueue.pop_front();

                if( i > 0 )
                    mAvail.Wait();
            }

            return( result );
        }
    }

    std::vector< T > PopAll()
    {
        MUTEX_SCOPE( mMutex );

        std::vector< T > result;

        if( !mQueue.empty() )
        {
            size_t count = mQueue.size();

            result.reserve( count );
            result.insert( result.end(), mQueue.begin(), mQueue.end() );

            mQueue.clear();

            while( count-- )
                mAvail.Wait();
        }

        return( result );
    }

    void Clear()
    {
        MUTEX_SCOPE( mMutex );

        size_t count = mQueue.size();

        mQueue.clear();
        mAvail.Wait(count);
    }
};


#endif // CORVID_THREAD_H__
