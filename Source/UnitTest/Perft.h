// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

const int MAX_PARALLEL_DEPTH = 5;

static void GatherPerftParallelPositions( const Position& pos, int depth, vector< Position >* dest )
{
    MoveList valid;
    valid.FindMoves( pos );

    for( int i = 0; i < valid._Count; i++ )
    {
        Position next = pos;
        next.Step( valid._Move[i] );

        if( depth == (MAX_PARALLEL_DEPTH + 1) )
            dest->push_back( next );
        else
            GatherPerftParallelPositions( next, depth - 1, dest );
    }
}

static u64 CalcPerftParallel( const Position& pos, int depth )
{
    vector< Position > positions( 16384 );
    GatherPerftParallelPositions( pos, depth, &positions );

    u64 total = 0;

    #pragma omp parallel for reduction(+: total) schedule(dynamic)
    for( int i = 0; i < (int) positions.size(); i++ )
    {
        u64 CalcPerftInternal( const Position & pos, int depth );
        u64 subtotal = CalcPerftInternal( positions[i], MAX_PARALLEL_DEPTH );
        total = total + subtotal;
    }

    return( total );
}

static u64 CalcPerftInternal( const Position& pos, int depth )
{
    if( (depth > MAX_PARALLEL_DEPTH) && (depth <= MAX_PARALLEL_DEPTH + 3) )
    {
        return( CalcPerftParallel( pos, depth ) );
    }

    MoveList valid;
    valid.FindMoves( pos );

    u64 total = 0;

    for( int i = 0; i < valid._Count; i++ )
    {
        Position next = pos;
        next.Step( valid._Move[i] );

        if( depth == 2 )
        {
            MoveList dummy;
            total += dummy.FindMoves( next );
        }

        else
        {
            total += CalcPerftInternal( next, depth - 1 );
        }
    }

    return( total );
}

static u64 CalcPerft( const Position& pos, int depth )
{
    if( depth < 2 )
    {
        MoveList dummy;
        return( dummy.FindMoves( pos ) );
    }

    return( CalcPerftInternal( pos, depth ) );
}

static map< MoveSpec, u64 > DividePerft( const Position& pos, int depth )
{
    MoveList valid;
    valid.FindMoves( pos );

    map< MoveSpec, u64 > result;
    for( int i = 0; i < valid._Count; i++ )
    {
        Position next = pos;
        next.Step( valid._Move[i] );

        u64 count = (depth > 1)? CalcPerft( next, depth - 1 ) : 1;
        result[valid._Move[i]] = count;
    }

    return result;  
}
