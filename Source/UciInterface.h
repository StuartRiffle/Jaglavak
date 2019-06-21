// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "TreeSearch/TreeSearch.h"

class UciInterface
{
    TreeSearch _TreeSearch;
    GlobalSettings* _Settings;
    bool _DebugMode;

public:
    UciInterface( GlobalSettings* settings );
    bool ProcessCommand( const char* cmd );
};

