// Random.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_RANDOM_H__
#define CORVID_RANDOM_H__

struct RandomGen
{
    u64 mState;

    RandomGen( u64 seed = 1 ) : mState( seed ) {}
    void SetSeed( u64 seed ) {  mState = seed; }

    u64 GetNext()
    {
        u64 n = mState;

        // 64-bit LCG using terms from Knuth

        n *= 6364136223846793005ULL;
        n += 1442695040888963407ULL;

        // 64-bit Wang mixing function

        n += ~(n << 32);
        n ^=  (n >> 22);
        n += ~(n << 13);
        n ^=  (n >>  8);
        n +=  (n <<  3);
        n ^=  (n >> 15);
        n += ~(n << 27);
        n ^=  (n >> 31);

        mState = n;
        return n;
    }

    u64 GetRange( u64 range )
    {
        u64 n = GetNext();
        return (n % range);
    }

    float GetFloat()
    {
        u64 n = GetNext() >> 32;
        return (n * 1.0f) / (1ULL << 32);
    }
};

#endif // CORVID_RANDOM_H__

