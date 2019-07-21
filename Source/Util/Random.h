// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

PDECL INLINE u64 WangMix64( u64 s )
{
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

    PDECL RandomGen( u64 seed = 1 ) { SetSeed( seed ); }
    PDECL void SetSeed( u64 seed ) { assert( seed != 0 ); s = seed; }

    PDECL u64 GetNext()
    {
        s = WangMix64( s );
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
