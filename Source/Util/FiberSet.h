#pragma once

#include "boost/fiber/all.hpp"

namespace bf = fibers;

typedef bf::fiber        Fiber;
typedef bf::fiber::id    FiberId;

class FiberSet
{
    list< Fiber > _Fibers;
    map< FiberId, u64 > _FiberIndex;
    //FiberId _MainFiber;

    bool _EnableTrace = true;

public:
    FiberSet()
    {
        //_MainFiber = this->GetCurrentFiber();

        this->Trace( "Constructor" );
    }

    int GetCount() const { return (int) _Fibers.size(); }

    template< typename TFUNC >
    void Spawn( TFUNC& fiberproc )
    {
        //assert( GetCurrentFiber() == _MainFiber );

        this->Trace( "Before spawn" );
        _Fibers.emplace_back( fiberproc );
        this->Trace( "After spawn" );
    }

    void YieldFiber()
    {
        this->Trace( "Before yield" );
        this_fiber::yield();
        this->Trace( "After yield" );
    }

    void Update()
    {
        YieldFiber();
        JoinCompletedFibers();
    }

    void TerminateAll()
    {
        this->Trace( "Terminating ALL fibers" );

        _Fibers.clear();
        _FiberIndex.clear();
    }

private:

    FiberId GetCurrentFiber() const
    {
        auto thisFiber = this_fiber::get_id();
        return thisFiber;
    }

    void JoinCompletedFibers()
    {
        this->Trace( "Before joins" );
        auto joinable = remove_if( _Fibers.begin(), _Fibers.end(),
            []( Fiber& f ) -> bool { return f.joinable(); } );

        for( auto iter = joinable; iter != _Fibers.end(); ++iter )
        {
            this->Trace( "Joining" );
            iter->join();

            auto iterIndex = _FiberIndex.find( iter->get_id() );
            assert( iterIndex != _FiberIndex.end() );
            _FiberIndex.erase( iterIndex );
        }

        _Fibers.erase( joinable, _Fibers.end() );
        this->Trace( "After joins" );
    }

    void Trace( const char* msg )
    {
#if DEBUG
        if( !_EnableTrace )
            return;

        FiberId thisFiber = GetCurrentFiber();

        cout << "Fiber " << thisFiber << " ";
        auto iterIndex = _FiberIndex.find( thisFiber );
        if( iterIndex != _FiberIndex.end() )
        {
            u64 index = iterIndex->second;

            cout << "[" << index << "] ";
        }

        cout << msg << endl;
;
#endif
    }

};

