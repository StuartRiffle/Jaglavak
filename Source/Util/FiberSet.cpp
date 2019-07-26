// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "FiberSet.h"
using namespace boost;

FiberSet::~FiberSet()
{
    TerminateAll();
}

void FiberSet::YieldFiber()
{
    this_fiber::yield();
    _NumYields++;

    if( _TerminatingFibers )
        throw FiberUnwindException();
}

void FiberSet::UpdateAll()
{
    YieldFiber();
    DestroyCompletedFibers();
    _NumUpdates++;
}

void FiberSet::TerminateAll()
{
    _TerminatingFibers = true;
    this_fiber::yield();
    _TerminatingFibers = false;

    DestroyCompletedFibers();
}

void FiberSet::DestroyCompletedFibers()
{
    auto iter = _Fibers.begin();
    while( iter != _Fibers.end() )
    {
        auto next = iter;
        ++next;

        if( iter->_Done )
        {
            iter->_Fiber->join();
            _NumJoins++;

            _Fibers.erase( iter );
        }

        iter = next;
    }
}

void FiberSet::Trace( const char* msg )
{
#if DEBUG
    cout << "Fiber " << this_fiber::get_id() << " " << msg << endl;
#endif
}



