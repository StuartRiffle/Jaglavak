// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct ScoreCard
{
    atomic64_t mWins[2];
    atomic64_t mPlays;

    PDECL ScoreCard() { this->Clear(); }

    PDECL ScoreCard( const ScoreCard& rhs ) 
    {
        mWins[BLACK] = (u64) rhs.mWins[BLACK];
        mWins[WHITE] = (u64) rhs.mWins[WHITE];
        mPlays = (u64) rhs.mPlays;
    }

    PDECL void Clear()
    {
        mWins[BLACK] = 0;
        mWins[WHITE] = 0;
        mPlays = 0;
    }

    PDECL ScoreCard& operator=(const ScoreCard& sc)
    {
        PlatStoreAtomic( &mWins[BLACK], sc.mWins[BLACK] );
        PlatStoreAtomic( &mWins[WHITE], sc.mWins[WHITE] );
        PlatStoreAtomic( &mPlays, sc.mPlays );
        return *this;
    }

    PDECL void Add( const ScoreCard& sc )
    {
        PlatAddAtomic( &mWins[BLACK], sc.mWins[BLACK] );
        PlatAddAtomic( &mWins[WHITE], sc.mWins[WHITE] );
        PlatAddAtomic( &mPlays, sc.mPlays );
    }
};
