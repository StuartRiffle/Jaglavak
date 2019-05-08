// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct UciOptionInfo
{
    int         mIndex;
    const char* mName;
    int         mMin;
    int         mMax;
    int         mDefault;
};


class TreeSearch;

class UciEngine
{
    unique_ptr< TreeSearch >   mSearcher;
    GlobalOptions   mOptions;
    bool            mDebugMode;

public:
    UciEngine();

    const UciOptionInfo* GetOptionInfo();
    void SetDefaultOptions();
    void SetOptionByName( const char* name, int value );
    bool ProcessCommand( const char* cmd );
};

