// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class GlobalSettings
{
    unordered_map< string, int > _Value;

public:
    int& operator[]( const string& key ) { return _Value[str]; }

    void Initialize( const vector< string >& configFiles );
    void LoadJsonValues( string json );
};


