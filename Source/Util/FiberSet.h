#pragma once

#include "boost/fiber/all.hpp"
typedef public boost::fibers::fiber Fiber;

struct FiberInstance
{
    unique_ptr< Fiber > _Fiber;
    bool _Done = false;

    template< typename TFUNC >
    void Spawn( TFUNC& fiberproc )
    {
        _Fiber.reset( new Fiber( [=]()
            {
                fiberproc(); 
                _Done = true;

            } ) );
    }
};

class FiberSet
{
    list< FiberInstance > _Fibers;

    u64 _NumUpdates = 0;
    u64 _NumYields  = 0;
    u64 _NumSpawns  = 0;
    u64 _NumJoins   = 0;

public:
    template< typename TFUNC >
    void Spawn( TFUNC& fiberproc )
    {
        _Fibers.emplace_back();
        _Fibers.back().Spawn( fiberproc );
        _NumSpawns++;
    }

    void YieldFiber();
    void UpdateAll();
    void TerminateAll();

    int GetCount() const { return (int) _Fibers.size(); }

private:
    void DestroyCompletedFibers();
    void Trace( const char* msg );
};



