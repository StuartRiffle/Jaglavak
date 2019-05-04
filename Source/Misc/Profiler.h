// Profiler.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle


enum
{
    MAX_BRANCHES = 8,
};

struct ProfilerNode
{
    int mCount;
    u64 mHash[MAX_BRANCHES];
    const char* mDesc[MAX_BRANCHES];
    unique_ptr< ProfilerNode > mLink[MAX_BRANCHES];

    u64 mStartTick;
    u64 mTotalTicks;
    u64 mTotalEntries;

    ProfilerNode()
    {
        mCount = 0;
        mStartTick = 0;
        mTotalTicks = 0;
        mTotalEntries = 0;
    }

    INLINE void Enter()
    {
        mTotalEntries++;
        mStartTick = PlatGetClockTick();
    }

    INLINE void Leave()
    {
        u64 stopTick = PlatGetClockTick();
        u64 elapsed = stopTick - mStartTick;

        mTotalTicks += elapsed;
    }
};

struct Profiler
{
    ProfilerNode mRoot;
    std::vector< ProfilerNode* > mStack;

    Profiler()
    {
        mStack.push_back( &mRoot );
    }

    void Push( const char* desc )
    {
        assert( !mStack.empty() );
        ProfilerNode node = *mStack.last();

        for( int i = 0; i < MAX_BRANCHES; i++ )
        {
            if( mDesc[i] == desc )
            {
                mStack.push_back( mLink[i] );
                mLink[i]->Enter();
                return;
            }
        }

        assert( node->mCount < MAX_BRANCHES );
        if( node->mCount == MAX_BRANCHES )
            return;

        int idx = node->mCount++;

        mDesc[idx] = desc;
        mLink[idx] = new ProfilerNode();

        mStack.push_back( mLink[idx] );
        mLink[idx]->Enter();
    }

    void Pop()
    {
        ProfilerNode* node = mStack.last();
        node->Leave();

        mStack.pop_back();
    }
};

thread_local Profiler gThreadProfiler;

struct ProfilerScope
{
    INLINE ProfilerScope( const char* str ) 
    {
        gThreadProfiler.Push( str );
    }

    INLINE ~ProfilerScope()
    {
        gThreadProfiler.Pop();
    }
};

#define PROFILER_SCOPE ProfilerScope




