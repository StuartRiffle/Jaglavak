#pragma once

#include "boost/fiber/all.hpp"

class FiberSet
{
    list< boost::fibers::fiber > _Fibers;

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

            boost::fibers::fiber& fiber = *iter;
            if( fiber.joinable() )
            {
                fiber.join();
                _Fibers.erase( iter );
            }

            iter = next;
        }
    }
};

#define FIBER_YIELD() boost::this_fiber::yield()
