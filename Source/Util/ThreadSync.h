// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct Semaphore
{
#if TOOLCHAIN_GCC
    sem_t       mHandle;
    Semaphore()  { sem_init( &mHandle, 0, 0 ); }
    ~Semaphore() { sem_destroy( &mHandle ); }
    void Post( int count = 1 )  { while( count-- ) sem_post( &mHandle ); }
    void Wait() { while( sem_wait( &mHandle ) ); }
#elif TOOLCHAIN_MSVC
    HANDLE      mHandle;
    Semaphore() : mHandle( CreateSemaphore( NULL, 0, LONG_MAX, NULL ) ) {}
    ~Semaphore() { CloseHandle( mHandle ); }
    void Post() { ReleaseSemaphore( mHandle, 1, NULL ); }
    void Wait() { WaitForSingleObject( mHandle, INFINITE ); }
#endif
};
