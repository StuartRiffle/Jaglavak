// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class GlobalSettings
{
    unordered_map< string, int > _Value;

public:
    int operator[]( const string& key ) const
    {
        return _Value[str];
    }

    void Set( const string& key, int val )
    {
        _Value[key] = val;
    }

    void Initialize( const vector< string >& configFiles );
    void LoadJsonValues( string json );
};


