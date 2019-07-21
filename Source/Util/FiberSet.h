#pragma once

#include "boost/fiber/all.hpp"
typedef public boost::fibers::fiber Fiber;

struct FiberUnwindException : public std::exception {};

struct FiberInstance
{
    unique_ptr< Fiber > _Fiber;
    bool _Done = false;

    template< typename TFUNC >
    void Spawn( TFUNC& fiberproc )
    {
        auto wrapper = 
            [=]()
            {
                try
                {
                    fiberproc();
                }
                catch( FiberUnwindException& )
                {
                    // This is how the fibers are terminated
                }

                _Done = true;
            };

        _Fiber.reset( new Fiber( wrapper ) );
    }
};

class FiberSet
{
    list< FiberInstance > _Fibers;
    bool _TerminatingFibers = false;

    u64 _NumUpdates = 0;
    u64 _NumYields  = 0;
    u64 _NumSpawns  = 0;
    u64 _NumJoins   = 0;

public:
    ~FiberSet();

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



