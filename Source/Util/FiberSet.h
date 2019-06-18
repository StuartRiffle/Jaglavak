#pragma once

#include "boost/fiber/all.hpp"

typedef boost::fibers::fiber Fiber;

class FiberSet
{
    list< Fiber > _Fibers;

public:

    template< typename T >
    void Spawn( T& func )
    {
        _Fibers.emplace_back( [=]() { func(); } );
    }

    void Update()
    {
        boost::this_fiber::yield();
        this->DiscardJoinableFibers();
    }

    int GetSize()
    {
        return (int) _Fibers.size();
    }

private:

    void DiscardJoinableFibers()
    {
        auto iter = _Fibers.begin();
        while( iter != _Fibers.end() )
        {
            auto next = iter;
            ++next;

            Fiber& fiber = *iter;
            if( fiber.joinable() )
            {
                fiber.join();
                _Fibers.erase( iter );
            }

            iter = next;
        }
    }
};

#define YIELD_FIBER() boost::this_fiber::yield()
