// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct ScoreCard
{
    atomic64_t _Wins[2];
    atomic64_t _Plays;

    PDECL ScoreCard() { this->Clear(); }

    PDECL ScoreCard( const ScoreCard& rhs ) 
    {
        _Wins[BLACK] = (u64) rhs._Wins[BLACK];
        _Wins[WHITE] = (u64) rhs._Wins[WHITE];
        _Plays = (u64) rhs._Plays;
    }

    PDECL void Clear()
    {
        _Wins[BLACK] = 0;
        _Wins[WHITE] = 0;
        _Plays = 0;
    }

    PDECL ScoreCard& operator=(const ScoreCard& sc)
    {
        PlatStoreAtomic( &_Wins[BLACK], sc._Wins[BLACK] );
        PlatStoreAtomic( &_Wins[WHITE], sc._Wins[WHITE] );
        PlatStoreAtomic( &_Plays, sc._Plays );
        return *this;
    }

    PDECL void Add( const ScoreCard& sc )
    {
        PlatAddAtomic( &_Wins[BLACK], sc._Wins[BLACK] );
        PlatAddAtomic( &_Wins[WHITE], sc._Wins[WHITE] );
        PlatAddAtomic( &_Plays, sc._Plays );
    }
};
