                                       
struct PlayoutJobInfo
{
    Position            mPosition;
    PlayoutOptions      mOptions;
    int                 mNumGames;
    MoveList            mPathFromRoot;
};

struct PlayoutJobResult
{
    u64         mPositionHash;
    MoveList    mPathFromRoot;
    ScoreCard   mScores;
};

typedef std::shared_ptr< PlayoutJobInfo >       PlayoutJobInfoRef;
typedef std::shared_ptr< PlayoutJobResult >     PlayoutJobResultRef;

typedef ThreadSafeQueue< PlayoutJobInfoRef >    PlayoutJobQueue;
typedef ThreadSafeQueue< PlayoutJobResultRef >  PlayoutResultQueue;


