// Random.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef JAGLAVAK_RANDOM_H__
#define JAGLAVAK_RANDOM_H__

struct RandomGen
{
    u64 s;

    PDECL RandomGen( u64 seed = 1 ) : s( seed ) {}
    PDECL void SetSeed( u64 seed ) {  s = seed; }

    PDECL u64 GetNext()
    {
        // 64-bit LCG using terms from Knuth

        s *= 6364136223846793005ULL;
        s += 1442695040888963407ULL;

        // Thomas Wang's 64-bit mix function

        s = ~s + (s << 21);
        s =  s ^ (s >> 24);
        s =  s + (s << 3) + (s << 8);
        s =  s ^ (s >> 14);
        s =  s + (s << 2) + (s << 4);
        s =  s ^ (s >> 28);
        s =  s + (s << 31);

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

#endif // JAGLAVAK_RANDOM_H__

