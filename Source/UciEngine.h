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

struct UciSearchConfig
{
    int                 mWhiteTimeLeft;   
    int                 mBlackTimeLeft;   
    int                 mWhiteTimeInc;    
    int                 mBlackTimeInc;    
    int                 mTimeControlMoves;
    int                 mMateSearchDepth; 
    int                 mDepthLimit;       
    int                 mNodesLimit;       
    int                 mTimeLimit; 
    MoveList            mLimitMoves;

    UciSearchConfig()   { this->Clear(); }
    void Clear()        { memset( this, 0, sizeof( *this ) ); }
};

class TreeSearch;

class UciEngine
{
    PTR< TreeSearch >   mSearcher;
    GlobalOptions   mOptions;
    bool            mDebugMode;

public:
    UciEngine();

    const UciOptionInfo* GetOptionInfo();
    void SetDefaultOptions();
    void SetOptionByName( const char* name, int value );
    bool ProcessCommand( const char* cmd );
};

