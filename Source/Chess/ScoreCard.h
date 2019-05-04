
struct ScoreCard
{
    u64 mWins[2];
    u64 mPlays;

    PDECL ScoreCard()
    {
        this->Clear();
    }

    PDECL void Clear()
    {
        mWins[BLACK] = 0;
        mWins[WHITE] = 0;
        mPlays = 0;
    }

    PDECL ScoreCard& operator+=( const ScoreCard& sc )
    {
        mWins[BLACK] += sc.mWins[BLACK];
        mWins[WHITE] += sc.mWins[WHITE];
        mPlays += sc.mPlays;
        return *this;
    }
};
