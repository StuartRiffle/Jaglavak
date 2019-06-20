// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class GlobalSettings
{
    unordered_map< hash_t, int > _Values;

public:
    int operator[]( const char* key ) const
    {
        hash_t hash = HashString( key );
        return _Values[hash];
    }

    void Set( const char* key, int val )
    {
        hash_t hash = HashString( key );
        _Values[hash] = val;
    }
};


