// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "FiberSet.h"
using namespace boost;

void FiberSet::YieldFiber()
{
    this_fiber::yield();
}

void FiberSet::UpdateAll()
{
    this_fiber::yield();
    DestroyCompletedFibers();
}

void FiberSet::TerminateAll()
{
    _Fibers.clear();
}

void FiberSet::DestroyCompletedFibers()
{
    auto completed = remove_if( _Fibers.begin(), _Fibers.end(),
        []( FiberInstance& f ) -> bool { return f._Done; } );

    for( auto iter = completed; iter != _Fibers.end(); ++iter )
        iter->_Fiber->join();

    _Fibers.erase( completed, _Fibers.end() );
}

void FiberSet::Trace( const char* msg )
{
#if DEBUG
    cout << "Fiber " << this_fiber::get_id() << " " << msg << endl;
#endif
}


