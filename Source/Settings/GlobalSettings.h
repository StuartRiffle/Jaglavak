// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct Metrics
{
    u64 _NodesExpanded;
    u64 _GamesPlayed;
    u64 _PositionsPlayed;
    u64 _BatchesQueued;
    u64 _BatchesDone;
    u64 _BatchTotalLatency;
    u64 _BatchTotalRuntime;

    Metrics() { Clear(); }
    void Clear() { memset( this, 0, sizeof( *this ) ); }

    void operator+=( const Metrics& rhs )
    {
        const u64* src  = (u64*) &rhs;
        u64* dest = (u64*) this;
        int count = (int) (sizeof( *this ) / sizeof( u64 ));

        for( int i = 0; i < count; i++ )
            dest[i] += src[i];
    }
};

class GlobalSettings
{
    map< string, double > _Value;

public:
    template< typename T = int >
    T Get( const string& key ) const
    { 
        T value = 0;

        auto iter = _Value.find( key );
        if( iter != _Value.end() )
            value = ( T) iter->second;
        else
            cout << "WARNING: " << key << " is undefined" << endl;

        return value;
    }

    void Set( const string& key, double val ) 
    { 
        _Value[key] = val; 
    }

    void Initialize( const vector< string >& configFiles );
    void LoadJsonValues( const string& json );
    void PrintListForUci() const;

public:
    // FIXME


};


