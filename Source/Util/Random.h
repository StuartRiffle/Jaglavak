// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

PDECL INLINE u64 Mix64( u64 s )
{
    // This is Thomas Wang's 64-bit mix function

    s = ~s + (s << 21);
    s =  s ^ (s >> 24);
    s =  s + (s << 3) + (s << 8);
    s =  s ^ (s >> 14);
    s =  s + (s << 2) + (s << 4);
    s =  s ^ (s >> 28);
    s =  s + (s << 31);

    return s;
}

struct RandomGen
{
    u64 s;

    PDECL RandomGen( u64 seed = 1 ) : { SetSeed( seed ) }
    PDECL void SetSeed( u64 seed ) { assert( seed != 0 ); s = seed; }

    PDECL u64 GetNext()
    {
        s = Mix64( s );
        return s;
    }

    PDECL u64 GetRange( u64 range )
    {
        return (GetNext() % range);
    }

    PDECL float GetFloat()
    {
        return ((GetNext() >> 32) * 1.0f) / (1ULL << 32);
    }
};



typedef uint64_t hash_t;

static extern hash_t StringHash( const char* str )
{
    // FNV-1a hash with normal parameters

    hash_t hash = 14695981039346656037ULL;
    while( *str )
        hash = (hash * 1099511628211) ^ *str++;

    // A final mix for peace of mind

    hash = Mix64( hash );
    return hash;
}
