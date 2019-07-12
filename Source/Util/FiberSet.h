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

public:
    template< typename TFUNC >
    void Spawn( TFUNC& fiberproc )
    {
        _Fibers.emplace_back();
        _Fibers.back().Spawn( fiberproc );
    }

    void YieldFiber();
    void UpdateAll();
    void TerminateAll();

    int GetCount() const { return (int) _Fibers.size(); }

private:
    void DestroyCompletedFibers();
    void Trace( const char* msg );
};



