// Evaluation.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_EVAL_H__
#define CORVID_EVAL_H__

enum
{
    EVAL_PAWNS,              
    EVAL_CENTER_PAWNS,       
    EVAL_CHAINED_PAWNS,      
    EVAL_PASSED_PAWNS,       
    EVAL_PAWNS_GUARD_KING,   
    EVAL_PROMOTING_SOON,     
    EVAL_PROMOTING_IMMED,    
    EVAL_KNIGHTS,            
    EVAL_KNIGHTS_DEVEL,      
    EVAL_KNIGHTS_FIRST,      
    EVAL_KNIGHTS_INTERIOR,   
    EVAL_KNIGHTS_CENTRAL,    
    EVAL_BISHOPS,            
    EVAL_BISHOPS_DEVEL,      
    EVAL_BOTH_BISHOPS,       
    EVAL_BISHOPS_INTERIOR,   
    EVAL_BISHOPS_CENTRAL,    
    EVAL_ROOKS,              
    EVAL_ROOKS_DEVEL,        
    EVAL_ROOK_ON_RANK_7,     
    EVAL_ROOKS_CONNECTED,   
    EVAL_ROOKS_OPEN_FILE,    
    EVAL_QUEENS,             
    EVAL_QUEEN_DEVEL,        
    EVAL_QUEENS_INTERIOR,    
    EVAL_QUEENS_CENTRAL,     
    EVAL_KINGS,              
    EVAL_KING_CASTLED,      
    EVAL_MOBILITY,           
    EVAL_ATTACKING,          
    EVAL_DEFENDING,          
    EVAL_ENEMY_TERRITORY,   
    EVAL_CENTER_PIECES,      
    EVAL_CENTER_CONTROL,     

    NUM_EVAL_TERMS
};

struct EvalWeightSet
{
    EvalWeight mWeights[NUM_EVAL_TERMS];
};

struct EvalWeightBlendInfo
{
    float   mOpening;
    float   mMidgame;
    float   mEndgame;
};

struct Evaluation
{
    PDECL static EvalWeightBlendInfo* GetDefaultWeightInfo()
    {
        static EvalWeightBlendInfo sDefaultWeights[] =
        {
            {   100.00f,    100.00f,    100.00f },  // EVAL_PAWNS                   
            {    10.00f,      0.00f,      0.00f },  // EVAL_CENTER_PAWNS            
            {    10.00f,      2.75f,      2.75f },  // EVAL_CHAINED_PAWNS           
            {     0.01f,      0.01f,      0.01f },  // EVAL_PASSED_PAWNS            
            {    10.00f,      5.62f,      5.62f },  // EVAL_PAWNS_GUARD_KING        
            {    32.98f,     32.98f,     50.00f },  // EVAL_PROMOTING_SOON          
            {    80.09f,     80.09f,    100.00f },  // EVAL_PROMOTING_IMMED         
            {   262.62f,    262.62f,    262.62f },  // EVAL_KNIGHTS                 
            {    15.00f,      0.00f,      0.00f },  // EVAL_KNIGHTS_DEVEL           
            {    10.00f,      0.00f,      0.00f },  // EVAL_KNIGHTS_FIRST           
            {    15.10f,     15.10f,     15.10f },  // EVAL_KNIGHTS_INTERIOR        
            {    20.17f,     20.17f,     20.17f },  // EVAL_KNIGHTS_CENTRAL         
            {   282.00f,    282.00f,    282.00f },  // EVAL_BISHOPS                 
            {    10.00f,      0.00f,      0.00f },  // EVAL_BISHOPS_DEVEL           
            {    33.25f,     33.25f,     33.25f },  // EVAL_BOTH_BISHOPS            
            {     6.26f,      6.26f,      6.26f },  // EVAL_BISHOPS_INTERIOR        
            {    16.72f,     16.72f,     16.72f },  // EVAL_BISHOPS_CENTRAL         
            {   453.00f,    453.00f,    453.00f },  // EVAL_ROOKS                   
            {    10.00f,      0.00f,      0.00f },  // EVAL_ROOKS_DEVEL             
            {    15.69f,     15.69f,     15.69f },  // EVAL_ROOK_ON_RANK_7          
            {     0.00f,      0.00f,      0.00f },  // EVAL_ROOKS_CONNECTED         
            {    33.30f,     33.30f,     33.30f },  // EVAL_ROOKS_OPEN_FILE         
            {   920.00f,    920.00f,    920.00f },  // EVAL_QUEENS                  
            {    10.00f,      0.00f,      0.00f },  // EVAL_QUEEN_DEVEL             
            {     4.57f,      4.57f,      4.57f },  // EVAL_QUEENS_INTERIOR         
            {     8.72f,      8.72f,      8.72f },  // EVAL_QUEENS_CENTRAL          
            { 20000.00f,  20000.00f,  20000.00f },  // EVAL_KINGS                   
            {    30.00f,     10.00f,      0.00f },  // EVAL_KING_CASTLED            
            {     2.13f,      2.13f,      2.13f },  // EVAL_MOBILITY                
            {     8.14f,      8.14f,      8.14f },  // EVAL_ATTACKING               
            {     0.02f,      0.02f,      0.02f },  // EVAL_DEFENDING               
            {    10.00f,     24.09f,     10.00f },  // EVAL_ENEMY_TERRITORY         
            {    10.00f,      0.00f,      0.00f },  // EVAL_CENTER_PIECES           
            {     5.00f,      1.69f,      0.00f },  // EVAL_CENTER_CONTROL          
        };

        assert( (sizeof( sDefaultWeights ) / sizeof( EvalWeightBlendInfo )) == NUM_EVAL_TERMS );
        return sDefaultWeights;
    }

    PDECL static void BlendWeights( const EvalWeightBlendInfo* blendInfo, EvalWeightSet* dest, float gamePhase ) 
    {
        PROFILER_SCOPE( "Evaluation::BlendWeights" );

        float   openingPct  = 1 - Max( 0.0f, Min( 1.0f, gamePhase ) );
        float   endgamePct  = Max( 0.0f, Min( 1.0f, gamePhase - 1 ) );
        float   midgamePct  = 1 - (openingPct + endgamePct);

        for( int i = 0; i < NUM_EVAL_TERMS; i++ )
        {
            float termWeight =
                (blendInfo[i].mOpening * openingPct) +
                (blendInfo[i].mMidgame * midgamePct) +
                (blendInfo[i].mEndgame * endgamePct);

            dest->mWeights[i] = (EvalWeight) (termWeight * WEIGHT_SCALE);
        }
    }

    PDECL static void GenerateWeights( EvalWeightSet* dest, float gamePhase ) 
    {
        BlendWeights( GetDefaultWeightInfo(), dest, gamePhase );
    }

    template< typename SIMD >
    PDECL static SIMD ApplyWeights( const SIMD* eval, const EvalWeightSet& weights ) 
    {
        PROFILER_SCOPE( "Evaluation::ApplyWeights" );

        SIMD score = MulSigned32( eval[0], weights.mWeights[0] );
        for( int i = 1; i < NUM_EVAL_TERMS; i++ )
            score += MulSigned32( eval[i], weights.mWeights[i] );
        
        return( score >> WEIGHT_SHIFT );
    }

    PDECL static float CalcGamePhase( const Position& pos ) 
    {
        PROFILER_SCOPE( "Evaluation::CalcGamePhase" );

        // "openingness" starts at 1, then reduces to 0 over at most EVAL_OPENING_PLIES
        // "endingness" starts at 0, then increases as minor/major pieces are taken

        int     ply                 = pos.GetPlyZeroBased();
        int     whitePawnCount      = (int) CountBits( pos.mWhitePawns );
        int     whiteMinorCount     = (int) CountBits( pos.mWhiteKnights | pos.mWhiteBishops );
        int     whiteMajorCount     = (int) CountBits( pos.mWhiteRooks   | pos.mWhiteQueens );
        int     whitePieceCount     = whitePawnCount + whiteMinorCount + whiteMajorCount;
        int     blackPawnCount      = (int) CountBits( pos.mBlackPawns ); 
        int     blackMinorCount     = (int) CountBits( pos.mBlackKnights | pos.mBlackBishops );
        int     blackMajorCount     = (int) CountBits( pos.mBlackRooks   | pos.mBlackQueens );
        int     blackPieceCount     = blackPawnCount + blackMinorCount + blackMajorCount;
        int     lowestPieceCount    = Min( whitePieceCount, blackPieceCount );
        float   fightingSpirit      = lowestPieceCount / 15.0f; // (king not counted)
        float   openingness         = Max( 0.0f, fightingSpirit - (ply * 1.0f / EVAL_OPENING_PLIES) );
        int     bigCaptures         = 14 - (blackMinorCount + blackMajorCount + whiteMinorCount + whiteMajorCount);
        float   endingness          = Max( 0.0f, bigCaptures / 14.0f );

        return( (openingness > 0)? (1 - openingness) : (1 + endingness) );
    }

    template< typename SIMD >
    PDECL static SIMD EvaluatePosition( const PositionT< SIMD >& pos, const MoveMapT< SIMD >& mmap, const EvalWeightSet& weights ) 
    {
        PROFILER_SCOPE( "Evaluation::EvaluatePosition" );

        SIMD    eval[NUM_EVAL_TERMS];   

        CalcEvalTerms< SIMD >( pos, mmap, eval );

        SIMD    evalScore           = ApplyWeights( eval, weights );
        SIMD    score               = evalScore;
        SIMD    moveTargets         = mmap.CalcMoveTargets();
        SIMD    inCheck             = mmap.IsInCheck();
        SIMD    mateFlavor          = SelectIfNotZero( inCheck, (SIMD) EVAL_CHECKMATE, (SIMD) EVAL_STALEMATE );
        SIMD    evalConsideringMate = SelectIfNotZero( moveTargets, score, mateFlavor );    

        return( evalConsideringMate );
    }

    template< typename SIMD >
    PDECL static SIMD EvaluatePosition( const PositionT< SIMD >& pos, const MoveMapT< SIMD >& mmap ) 
    {
        EvalWeightSet weights;

        float gamePhase = CalcGamePhase( pos );
        GenerateWeights( &weights, gamePhase );

        return( EvaluatePosition< SIMD >( pos, mmap, weights ) );
    }

    template< typename SIMD >
    PDECL static void CalcEvalTerms( const PositionT< SIMD >& pos, const MoveMapT< SIMD >& mmap, SIMD* eval )
    {
        PROFILER_SCOPE( "Evaluation::CalcEvalTerms" );

        PositionT< SIMD > flipped;
        flipped.FlipFrom( pos );

        SIMD    evalWhite[NUM_EVAL_TERMS];
        SIMD    evalBlack[NUM_EVAL_TERMS];

        CalcSideEval< SIMD >( pos,     mmap, evalWhite );
        CalcSideEval< SIMD >( flipped, mmap, evalBlack );

        for( int i = 0; i < NUM_EVAL_TERMS; i++ )
            eval[i] = evalWhite[i] - evalBlack[i];
    }

    template< typename SIMD >
    PDECL static void CalcSideEval( const PositionT< SIMD >& pos, const MoveMapT< SIMD >& mmap, SIMD* eval )
    {
        const SIMD& whitePawns      = pos.mWhitePawns;    
        const SIMD& whiteKnights    = pos.mWhiteKnights;  
        const SIMD& whiteBishops    = pos.mWhiteBishops;  
        const SIMD& whiteRooks      = pos.mWhiteRooks;    
        const SIMD& whiteQueens     = pos.mWhiteQueens;   
        const SIMD& whiteKing       = pos.mWhiteKing;     
        const SIMD& blackPawns      = pos.mBlackPawns;    
        const SIMD& blackKnights    = pos.mBlackKnights;  
        const SIMD& blackBishops    = pos.mBlackBishops;  
        const SIMD& blackRooks      = pos.mBlackRooks;    
        const SIMD& blackQueens     = pos.mBlackQueens;   
        const SIMD& blackKing       = pos.mBlackKing;     

        SIMD    whiteDiag           = whiteBishops | whiteQueens;
        SIMD    whiteOrtho          = whiteRooks | whiteQueens;
        SIMD    whitePieces         = whitePawns | whiteKnights | whiteBishops | whiteRooks | whiteQueens | whiteKing;
        SIMD    blackPieces         = blackPawns | blackKnights | blackBishops | blackRooks | blackQueens | blackKing;
        SIMD    allPieces           = blackPieces | whitePieces;
        SIMD    empty               = ~allPieces;
        SIMD    pawnsMobility       = StepN( whitePawns ) & empty;
        SIMD    pawnsChained        = (StepNW( whitePawns ) | StepSW( whitePawns ) | StepSE( whitePawns ) | StepNE( whitePawns )) & whitePawns;
        SIMD    pawnsControl        = MaskOut( StepNW( whitePawns ) | StepNE( whitePawns ), whitePieces );
        SIMD    knightsControl      = StepKnights( whiteKnights );
        SIMD    diagControl         = SlideIntoExDiag( whiteDiag, empty, allPieces );
        SIMD    orthoControl        = SlideIntoExDiag( whiteOrtho, empty, allPieces );
        SIMD    kingControl         = StepOut( whiteKing );
        SIMD    whiteControl        = pawnsControl | knightsControl | diagControl | orthoControl | kingControl;
        SIMD    whiteMobility       = whiteControl & empty;
        SIMD    whiteAttacking      = whiteControl & blackPieces;
        SIMD    whiteDefending      = whiteControl & whitePieces;
        SIMD    inEnemyTerritory    = whitePieces & (RANK_5 | RANK_6 | RANK_7 | RANK_8);
        SIMD    knightsDevel        = CountBits( whiteKnights & ~(SQUARE_B1 | SQUARE_G1) );
        SIMD    bishopsDevel        = CountBits( whiteBishops & ~(SQUARE_C1 | SQUARE_F1) );    

        eval[EVAL_PAWNS]            = CountBits( whitePawns );                                
        eval[EVAL_CENTER_PAWNS]     = CountBits( whitePawns & CENTER_SQUARES );             
        eval[EVAL_CHAINED_PAWNS]    = CountBits( pawnsChained );                              
        eval[EVAL_PASSED_PAWNS]     = CountBits( PropN( whitePawns, ~blackPawns ) & RANK_8 ); 
        eval[EVAL_PAWNS_GUARD_KING] = CountBits( whitePawns & (StepNW( whiteKing ) | StepN( whiteKing ) | StepNE( whiteKing )) );                                               
        eval[EVAL_PROMOTING_SOON]   = CountBits( whitePawns & RANK_6 );                     
        eval[EVAL_PROMOTING_IMMED]  = CountBits( whitePawns & RANK_7 );                     

        eval[EVAL_KNIGHTS]          = CountBits( whiteKnights );                              
        eval[EVAL_KNIGHTS_DEVEL]    = knightsDevel;                                                 
        eval[EVAL_KNIGHTS_INTERIOR] = CountBits( whiteKnights & ~EDGE_SQUARES );              
        eval[EVAL_KNIGHTS_CENTRAL]  = CountBits( whiteKnights & CENTER_SQUARES );              

        eval[EVAL_BISHOPS]          = CountBits( whiteBishops );                              
        eval[EVAL_BISHOPS_DEVEL]    = bishopsDevel;
        eval[EVAL_BISHOPS_INTERIOR] = CountBits( whiteBishops & ~EDGE_SQUARES );              
        eval[EVAL_BISHOPS_CENTRAL]  = CountBits( whiteBishops & CENTER_SQUARES );              
        eval[EVAL_BOTH_BISHOPS]     = SelectIfNotZero( whiteBishops & LIGHT_SQUARES, (SIMD) 1 ) & SelectIfNotZero( whiteBishops & DARK_SQUARES, (SIMD) 1 );                                                  

        eval[EVAL_ROOKS]            = CountBits( whiteRooks );                                
        eval[EVAL_ROOKS_DEVEL]      = CountBits( whiteRooks & ~(SQUARE_A1 | SQUARE_H1) );   
        eval[EVAL_ROOK_ON_RANK_7]   = CountBits( whiteRooks & RANK_7 );                       
        eval[EVAL_ROOKS_CONNECTED]  = CountBits( PropExOrtho( whiteRooks, empty ) & whiteRooks );                                               
        eval[EVAL_ROOKS_OPEN_FILE]  = CountBits( PropN( whiteRooks, empty ) & RANK_8 );       

        eval[EVAL_QUEENS]           = CountBits( whiteQueens );                               
        eval[EVAL_QUEEN_DEVEL]      = CountBits( whiteQueens & ~(SQUARE_D1) );               
        eval[EVAL_QUEENS_INTERIOR]  = CountBits( whiteQueens & ~EDGE_SQUARES );              
        eval[EVAL_QUEENS_CENTRAL]   = CountBits( whiteQueens & CENTER_SQUARES );              

        eval[EVAL_KINGS]            = CountBits( whiteKing );                                 
        eval[EVAL_KING_CASTLED]     = CountBits( whiteKing & RANK_1 & ~SQUARE_E1 );           

        eval[EVAL_KNIGHTS_FIRST]    = SubClampZero( knightsDevel, bishopsDevel );
        eval[EVAL_MOBILITY]         = CountBits( whiteMobility );                             
        eval[EVAL_ATTACKING]        = CountBits( whiteAttacking );                            
        eval[EVAL_DEFENDING]        = CountBits( whiteDefending );                            
        eval[EVAL_ENEMY_TERRITORY]  = CountBits( inEnemyTerritory );                          
        eval[EVAL_CENTER_PIECES]    = CountBits( whitePieces  & CENTER_SQUARES );             
        eval[EVAL_CENTER_CONTROL]   = CountBits( whiteControl & CENTER_SQUARES );   
    }
};

#endif // CORVID_EVAL_H__
