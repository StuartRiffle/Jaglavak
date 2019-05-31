// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

enum
{
    CHECKBOX = 1,
};

struct UciOptionInfo
{
    int         _Index;
    const char* _Name;
    int         _IsCheckbox;
    int         _Value;
};


struct TreeSearch;

class UciEngine
{
    unique_ptr< TreeSearch >   _Searcher;
    GlobalOptions   _Options;
    bool            _DebugMode;

public:
    UciEngine();

    const UciOptionInfo* GetOptionInfo();
    void SetDefaultOptions();
    void SetOptionByName( const char* name, int value );
    bool ProcessCommand( const char* cmd );
};

