// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

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
            value = (T) iter->second;

        return value;
    }

    void Set( const string& key, double val ) 
    { 
        _Value[key] = val; 
    }

    void Initialize( const vector< string >& configFiles );
    void LoadJsonValues( const string& json );
    void PrintListForUci() const;
};


