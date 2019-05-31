// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct Semaphore
{
#if TOOLCHAIN_GCC
    sem_t       _Handle;
    Semaphore()  { sem_init( &_Handle, 0, 0 ); }
    ~Semaphore() { sem_destroy( &_Handle ); }
    void Post( int count = 1 )  { while( count-- ) sem_post( &_Handle ); }
    void Wait() { while( sem_wait( &_Handle ) ); }
#elif TOOLCHAIN_MSVC
    HANDLE      _Handle;
    Semaphore() : _Handle( CreateSemaphore( NULL, 0, LONG_MAX, NULL ) ) {}
    ~Semaphore() { CloseHandle( _Handle ); }
    void Post() { ReleaseSemaphore( _Handle, 1, NULL ); }
    void Wait() { WaitForSingleObject( _Handle, INFINITE ); }
#endif
};
