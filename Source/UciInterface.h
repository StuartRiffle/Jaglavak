// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "TreeSearch/TreeSearch.h"

class UciInterface
{
    unique_ptr< TreeSearch > _TreeSearch;
    GlobalSettings* _Settings;
    bool _DebugMode;

public:
    UciInterface( GlobalSettings* settings );

    void SetOptionByName( const char* name, int value );
    bool ProcessCommand( const char* cmd );
};

