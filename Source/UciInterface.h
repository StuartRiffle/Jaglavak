// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

enum
{
    CHECKBOX = 1,
};



class TreeSearch;

class UciInterface
{
    unique_ptr< TreeSearch > _TreeSearch;
    GlobalOptions _Options;
    bool _DebugMode;

public:
    UciInterface();

    const UciOptionInfo* GetOptionInfo();
    void SetDefaultOptions();
    void SetOptionByName( const char* name, int value );
    bool ProcessCommand( const char* cmd );
};

