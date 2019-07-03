// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

/// A snapshot of the game state

template< typename SIMD >
struct ALIGN_SIMD PositionT
{
    SIMD        _WhitePawns;        /// Bitmask of the white pawns 
    SIMD        _WhiteKnights;      /// Bitmask of the white knights    
    SIMD        _WhiteBishops;      /// Bitmask of the white bishops    
    SIMD        _WhiteRooks;        /// Bitmask of the white rooks 
    SIMD        _WhiteQueens;       /// Bitmask of the white queens    
    SIMD        _WhiteKing;         /// Bitmask of the white king 
                                      
    SIMD        _BlackPawns;        /// Bitmask of the black pawns     
    SIMD        _BlackKnights;      /// Bitmask of the black knights
    SIMD        _BlackBishops;      /// Bitmask of the black bishops    
    SIMD        _BlackRooks;        /// Bitmask of the black rooks     
    SIMD        _BlackQueens;       /// Bitmask of the black queens    
    SIMD        _BlackKing;         /// Bitmask of the black king 
                                     
    SIMD        _CastlingAndEP;     /// Bitmask of EP capture targets and castling targets
    SIMD        _BoardFlipped;      /// A mask which is ~0 when this structure is white/black flipped from the actual position
    SIMD        _WhiteToMove;       /// 1 if it's white to play, 0 if black
    SIMD        _HalfmoveClock;     /// Number of halfmoves since the last capture or pawn move
    SIMD        _FullmoveNum;       /// Starts at 1, increments after black moves
    SIMD        _GameResult;        /// Set to RESULT_* when the game is over

    /// Reset the fields to the start-of-game position.
    
    PDECL void Reset()
    {
        _WhitePawns         = RANK_2;
        _WhiteKnights       = SQUARE_B1 | SQUARE_G1;
        _WhiteBishops       = SQUARE_C1 | SQUARE_F1;
        _WhiteRooks         = SQUARE_A1 | SQUARE_H1;
        _WhiteQueens        = SQUARE_D1;        
        _WhiteKing          = SQUARE_E1;

        _BlackPawns         = RANK_7;              
        _BlackKnights       = SQUARE_B8 | SQUARE_G8;
        _BlackBishops       = SQUARE_C8 | SQUARE_F8;
        _BlackRooks         = SQUARE_A8 | SQUARE_H8;
        _BlackQueens        = SQUARE_D8;
        _BlackKing          = SQUARE_E8;

        _CastlingAndEP      = SQUARE_A1 | SQUARE_H1 | SQUARE_A8 | SQUARE_H8;
        _BoardFlipped       = 0;
        _WhiteToMove        = 1;
        _HalfmoveClock      = 0;
        _FullmoveNum        = 1;
        _GameResult         = RESULT_UNKNOWN;
    }

    /// Duplicate member values across SIMD lanes

    template< typename SCALAR >
    PDECL void Broadcast( const PositionT< SCALAR >& src )
    {
        _WhitePawns         = src._WhitePawns;   
        _WhiteKnights       = src._WhiteKnights; 
        _WhiteBishops       = src._WhiteBishops; 
        _WhiteRooks         = src._WhiteRooks;   
        _WhiteQueens        = src._WhiteQueens;  
        _WhiteKing          = src._WhiteKing;
                            
        _BlackPawns         = src._BlackPawns;   
        _BlackKnights       = src._BlackKnights; 
        _BlackBishops       = src._BlackBishops; 
        _BlackRooks         = src._BlackRooks;   
        _BlackQueens        = src._BlackQueens;  
        _BlackKing          = src._BlackKing;

        _CastlingAndEP      = src._CastlingAndEP;
        _BoardFlipped       = src._BoardFlipped; 
        _WhiteToMove        = src._WhiteToMove;  
        _HalfmoveClock      = src._HalfmoveClock;
        _FullmoveNum        = src._FullmoveNum;  
        _GameResult         = src._GameResult;
    }


    /// Update the game state by applying a (presumably valid) move
    
    PDECL void Step( const MoveSpecT< SIMD >& move, MoveMapT< SIMD >* nextMapOut = NULL )
    {
        SIMD moveSrc    = SelectWithMask( _BoardFlipped,  FlipSquareIndex( move._Src ), move._Src );
        SIMD srcBit     = SquareBit( moveSrc );
        SIMD moveDest   = SelectWithMask( _BoardFlipped,  FlipSquareIndex( move._Dest ), move._Dest );
        SIMD destBit    = SquareBit( moveDest );

        SIMD isPawnMove = SelectIfNotZero( srcBit & _WhitePawns, MaskAllSet< SIMD >() );
        SIMD blackPieces= _BlackPawns | _BlackKnights | _BlackBishops | _BlackRooks | _BlackQueens;
        SIMD isCapture  = SelectIfNotZero( destBit & blackPieces, MaskAllSet< SIMD >() );

        // FIXME: isCapture not accounting for en passant?

        this->ApplyMove( move._Src, move._Dest, move._Type );
        this->FlipInPlace();

        _WhiteToMove   = SelectIfNotZero( _GameResult, _WhiteToMove,    _WhiteToMove ^ 1 );
        _FullmoveNum   = SelectIfNotZero( _GameResult, _FullmoveNum,    _FullmoveNum + _WhiteToMove );
        _HalfmoveClock = SelectIfNotZero( _GameResult, _HalfmoveClock, (_HalfmoveClock + 1) & ~(isPawnMove | isCapture) );

        MoveMapT< SIMD > localMap;
        MoveMapT< SIMD >* mmap = nextMapOut? nextMapOut : &localMap;

        this->CalcMoveMap( mmap );

        SIMD moveTargets = mmap->CalcMoveTargets();
        SIMD inCheck     = mmap->IsInCheck();
        SIMD win         = SelectIfNotZero( _WhiteToMove, (SIMD) RESULT_BLACK_WIN, (SIMD) RESULT_WHITE_WIN );
        SIMD winOrDraw   = SelectIfNotZero( inCheck, win, (SIMD) RESULT_DRAW );
        SIMD gameResult  = SelectIfZero( moveTargets, winOrDraw, (SIMD) RESULT_UNKNOWN );

        _GameResult = SelectIfNotZero( _GameResult, _GameResult, gameResult );
    }


    PDECL static INLINE SIMD CalcPieceType( const SIMD& knights, const SIMD& bishops, const SIMD& rooks, const SIMD& queens, const SIMD& king )
    {
        SIMD pieceType =
            SelectIfNotZero( knights,   (SIMD) KNIGHT ) |
            SelectIfNotZero( bishops,   (SIMD) BISHOP ) |
            SelectIfNotZero( rooks,     (SIMD) ROOK   ) |
            SelectIfNotZero( queens,    (SIMD) QUEEN  ) |
            SelectIfNotZero( king,      (SIMD) KING   );

        return( pieceType );
    }


    /// Update the piece positions 
      
    PDECL void ApplyMove( const SIMD& srcIdx, const SIMD& destIdx, const SIMD& moveType )  
    {
        SIMD    whitePawns          = _WhitePawns;    
        SIMD    whiteKnights        = _WhiteKnights;  
        SIMD    whiteBishops        = _WhiteBishops;  
        SIMD    whiteRooks          = _WhiteRooks;    
        SIMD    whiteQueens         = _WhiteQueens;   
        SIMD    whiteKing           = _WhiteKing;     
        SIMD    blackPawns          = _BlackPawns;    
        SIMD    blackKnights        = _BlackKnights;  
        SIMD    blackBishops        = _BlackBishops;  
        SIMD    blackRooks          = _BlackRooks;    
        SIMD    blackQueens         = _BlackQueens;   
        SIMD    blackKing           = _BlackKing;
        SIMD    castlingAndEP       = _CastlingAndEP;

        SIMD    moveSrc             = SelectWithMask( _BoardFlipped, FlipSquareIndex( srcIdx ),  srcIdx  );
        SIMD    moveDest            = SelectWithMask( _BoardFlipped, FlipSquareIndex( destIdx ), destIdx );
        SIMD    srcBit              = SquareBit( moveSrc );
        SIMD    destBit             = SquareBit( moveDest );

        SIMD    srcPawn             = srcBit & whitePawns;
        SIMD    srcKnight           = srcBit & whiteKnights;
        SIMD    srcBishop           = srcBit & whiteBishops;
        SIMD    srcRook             = srcBit & whiteRooks;
        SIMD    srcQueen            = srcBit & whiteQueens;
        SIMD    srcKing             = srcBit & whiteKing;

        SIMD    promotedKnight      = destBit & CmpEqual( moveType, (SIMD) PROMOTE_KNIGHT );
        SIMD    promotedBishop      = destBit & CmpEqual( moveType, (SIMD) PROMOTE_BISHOP );
        SIMD    promotedRook        = destBit & CmpEqual( moveType, (SIMD) PROMOTE_ROOK   );
        SIMD    promotedQueen       = destBit & CmpEqual( moveType, (SIMD) PROMOTE_QUEEN  );

        SIMD    formerPawn          = SelectIfNotZero( promotedKnight | promotedBishop | promotedRook | promotedQueen, srcPawn );
        SIMD    destPawn            = SelectIfNotZero( MaskOut( srcPawn, formerPawn ), destBit );
        SIMD    destKnight          = SelectIfNotZero( srcKnight | promotedKnight,     destBit );
        SIMD    destBishop          = SelectIfNotZero( srcBishop | promotedBishop,     destBit );
        SIMD    destRook            = SelectIfNotZero( srcRook   | promotedRook,       destBit );
        SIMD    destQueen           = SelectIfNotZero( srcQueen  | promotedQueen,      destBit );
        SIMD    destKing            = SelectIfNotZero( srcKing,                        destBit );

        SIMD    epTargetNext        = Shift< SHIFT_S >( Shift< SHIFT_N * 2 >( srcPawn ) & destPawn );
        SIMD    epVictimNow         = blackPawns & Shift< SHIFT_S >( destPawn & castlingAndEP & EP_SQUARES );
        SIMD    castleRookKing      = CmpEqual( (srcKing | destBit), (SIMD) (SQUARE_E1 | SQUARE_G1) ) & SQUARE_H1;
        SIMD    castleRookQueen     = CmpEqual( (srcKing | destBit), (SIMD) (SQUARE_E1 | SQUARE_C1) ) & SQUARE_A1;
        SIMD    disableCastleBit    = 0;                 

        srcRook                    |= castleRookKing | castleRookQueen;
        destRook                   |= SelectIfNotZero( castleRookKing,  (SIMD) SQUARE_F1 );
        destRook                   |= SelectIfNotZero( castleRookQueen, (SIMD) SQUARE_D1 );
        disableCastleBit           |= (srcRook & (SQUARE_A1 | SQUARE_H1));
        disableCastleBit           |= SelectIfNotZero( srcKing, (SIMD) (SQUARE_A1 | SQUARE_H1) );

        whitePawns                  = MaskOut( whitePawns,   srcPawn )   | destPawn;
        whiteKnights                = MaskOut( whiteKnights, srcKnight ) | destKnight;
        whiteBishops                = MaskOut( whiteBishops, srcBishop ) | destBishop;
        whiteRooks                  = MaskOut( whiteRooks,   srcRook )   | destRook;
        whiteQueens                 = MaskOut( whiteQueens,  srcQueen )  | destQueen;
        whiteKing                   = MaskOut( whiteKing,    srcKing )   | destKing;
        blackPawns                  = MaskOut( MaskOut( blackPawns, destBit ), epVictimNow );
        blackKnights                = MaskOut( blackKnights, destBit );
        blackBishops                = MaskOut( blackBishops, destBit );
        blackRooks                  = MaskOut( blackRooks,   destBit );
        blackQueens                 = MaskOut( blackQueens,  destBit );
        castlingAndEP               = MaskOut( castlingAndEP, disableCastleBit | EP_SQUARES ) | epTargetNext;

        _WhitePawns                 = SelectIfNotZero( _GameResult, _WhitePawns,    whitePawns    );
        _WhiteKnights               = SelectIfNotZero( _GameResult, _WhiteKnights,  whiteKnights  );
        _WhiteBishops               = SelectIfNotZero( _GameResult, _WhiteBishops,  whiteBishops  );
        _WhiteRooks                 = SelectIfNotZero( _GameResult, _WhiteRooks,    whiteRooks    );
        _WhiteQueens                = SelectIfNotZero( _GameResult, _WhiteQueens,   whiteQueens   );
        _WhiteKing                  = SelectIfNotZero( _GameResult, _WhiteKing,     whiteKing     );
        _BlackPawns                 = SelectIfNotZero( _GameResult, _BlackPawns,    blackPawns    );
        _BlackKnights               = SelectIfNotZero( _GameResult, _BlackKnights,  blackKnights  );
        _BlackBishops               = SelectIfNotZero( _GameResult, _BlackBishops,  blackBishops  );
        _BlackRooks                 = SelectIfNotZero( _GameResult, _BlackRooks,    blackRooks    );
        _BlackQueens                = SelectIfNotZero( _GameResult, _BlackQueens,   blackQueens   );
        _CastlingAndEP              = SelectIfNotZero( _GameResult, _CastlingAndEP, castlingAndEP );
    }


    /// Generate a map of valid moves from the current position
    
    PDECL void CalcMoveMap( MoveMapT< SIMD >* RESTRICT dest ) const
    {
        SIMD    whitePawns          = _WhitePawns;    
        SIMD    whiteKnights        = _WhiteKnights;  
        SIMD    whiteBishops        = _WhiteBishops;  
        SIMD    whiteRooks          = _WhiteRooks;    
        SIMD    whiteQueens         = _WhiteQueens;   
        SIMD    whiteKing           = _WhiteKing;     
        SIMD    blackPawns          = _BlackPawns;    
        SIMD    blackKnights        = _BlackKnights;  
        SIMD    blackBishops        = _BlackBishops;  
        SIMD    blackRooks          = _BlackRooks;    
        SIMD    blackQueens         = _BlackQueens;   
        SIMD    blackKing           = _BlackKing;     
        SIMD    castlingAndEP       = _CastlingAndEP;

        SIMD    whiteDiag           = whiteBishops | whiteQueens;
        SIMD    whiteOrtho          = whiteRooks | whiteQueens;
        SIMD    whitePieces         = whitePawns | whiteKnights | whiteBishops | whiteRooks | whiteQueens | whiteKing;
        SIMD    blackDiag           = blackBishops | blackQueens;
        SIMD    blackOrtho          = blackRooks | blackQueens;
        SIMD    blackPieces         = blackPawns | blackKnights | blackBishops | blackRooks | blackQueens | blackKing;
        SIMD    allPieces           = blackPieces | whitePieces;
        SIMD    empty               = ~allPieces;

        SIMD    kingViewN           = MaskOut( SlideIntoN(  SlideIntoN(  whiteKing, empty, whitePieces ), empty, blackOrtho ), whiteKing );
        SIMD    kingViewNW          = MaskOut( SlideIntoNW( SlideIntoNW( whiteKing, empty, whitePieces ), empty, blackDiag  ), whiteKing );
        SIMD    kingViewW           = MaskOut( SlideIntoW(  SlideIntoW(  whiteKing, empty, whitePieces ), empty, blackOrtho ), whiteKing );
        SIMD    kingViewSW          = MaskOut( SlideIntoSW( SlideIntoSW( whiteKing, empty, whitePieces ), empty, blackDiag  ), whiteKing );
        SIMD    kingViewS           = MaskOut( SlideIntoS(  SlideIntoS(  whiteKing, empty, whitePieces ), empty, blackOrtho ), whiteKing );
        SIMD    kingViewSE          = MaskOut( SlideIntoSE( SlideIntoSE( whiteKing, empty, whitePieces ), empty, blackDiag  ), whiteKing );
        SIMD    kingViewE           = MaskOut( SlideIntoE(  SlideIntoE(  whiteKing, empty, whitePieces ), empty, blackOrtho ), whiteKing );
        SIMD    kingViewNE          = MaskOut( SlideIntoNE( SlideIntoNE( whiteKing, empty, whitePieces ), empty, blackDiag  ), whiteKing );

        SIMD    kingDangerN         = SelectIfNotZero( (kingViewN  & blackPieces), kingViewN  );
        SIMD    kingDangerNW        = SelectIfNotZero( (kingViewNW & blackPieces), kingViewNW ) | StepNW( whiteKing, blackPawns );
        SIMD    kingDangerW         = SelectIfNotZero( (kingViewW  & blackPieces), kingViewW  );
        SIMD    kingDangerSW        = SelectIfNotZero( (kingViewSW & blackPieces), kingViewSW );
        SIMD    kingDangerS         = SelectIfNotZero( (kingViewS  & blackPieces), kingViewS  );
        SIMD    kingDangerSE        = SelectIfNotZero( (kingViewSE & blackPieces), kingViewSE );
        SIMD    kingDangerE         = SelectIfNotZero( (kingViewE  & blackPieces), kingViewE  );
        SIMD    kingDangerNE        = SelectIfNotZero( (kingViewNE & blackPieces), kingViewNE ) | StepNE( whiteKing, blackPawns );
        SIMD    kingDangerKnights   = StepKnights( whiteKing, blackKnights );

        SIMD    pinnedLineN         = SelectIfNotZero( (kingDangerN  & whitePieces), kingDangerN  );
        SIMD    pinnedLineNW        = SelectIfNotZero( (kingDangerNW & whitePieces), kingDangerNW );
        SIMD    pinnedLineW         = SelectIfNotZero( (kingDangerW  & whitePieces), kingDangerW  );
        SIMD    pinnedLineSW        = SelectIfNotZero( (kingDangerSW & whitePieces), kingDangerSW );
        SIMD    pinnedLineS         = SelectIfNotZero( (kingDangerS  & whitePieces), kingDangerS  );
        SIMD    pinnedLineSE        = SelectIfNotZero( (kingDangerSE & whitePieces), kingDangerSE );
        SIMD    pinnedLineE         = SelectIfNotZero( (kingDangerE  & whitePieces), kingDangerE  );
        SIMD    pinnedLineNE        = SelectIfNotZero( (kingDangerNE & whitePieces), kingDangerNE );
        SIMD    pinnedNS            = pinnedLineN  | pinnedLineS; 
        SIMD    pinnedNWSE          = pinnedLineNW | pinnedLineSE;
        SIMD    pinnedWE            = pinnedLineW  | pinnedLineE; 
        SIMD    pinnedSWNE          = pinnedLineSW | pinnedLineNE;
        SIMD    notPinned           = ~(pinnedNS | pinnedNWSE | pinnedWE | pinnedSWNE);

        SIMD    maskAllSet          = MaskAllSet< SIMD >();
        SIMD    checkMaskN          = SelectIfNotZero( (kingDangerN  ^ pinnedLineN ), kingDangerN , maskAllSet );
        SIMD    checkMaskNW         = SelectIfNotZero( (kingDangerNW ^ pinnedLineNW), kingDangerNW, maskAllSet );
        SIMD    checkMaskW          = SelectIfNotZero( (kingDangerW  ^ pinnedLineW ), kingDangerW , maskAllSet );
        SIMD    checkMaskSW         = SelectIfNotZero( (kingDangerSW ^ pinnedLineSW), kingDangerSW, maskAllSet );
        SIMD    checkMaskS          = SelectIfNotZero( (kingDangerS  ^ pinnedLineS ), kingDangerS , maskAllSet );
        SIMD    checkMaskSE         = SelectIfNotZero( (kingDangerSE ^ pinnedLineSE), kingDangerSE, maskAllSet );
        SIMD    checkMaskE          = SelectIfNotZero( (kingDangerE  ^ pinnedLineE ), kingDangerE , maskAllSet );
        SIMD    checkMaskNE         = SelectIfNotZero( (kingDangerNE ^ pinnedLineNE), kingDangerNE, maskAllSet );
        SIMD    checkMaskKnights    = SelectIfNotZero( kingDangerKnights, kingDangerKnights, maskAllSet );
        SIMD    checkMask           = checkMaskN & checkMaskNW & checkMaskW & checkMaskSW & checkMaskS & checkMaskSE & checkMaskE & checkMaskNE & checkMaskKnights;

        SIMD    slidingMovesN       = MaskOut( SlideIntoN(  whiteOrtho & (notPinned | pinnedNS  ), empty, blackPieces ), whiteOrtho );
        SIMD    slidingMovesNW      = MaskOut( SlideIntoNW( whiteDiag  & (notPinned | pinnedNWSE), empty, blackPieces ), whiteDiag  );
        SIMD    slidingMovesW       = MaskOut( SlideIntoW(  whiteOrtho & (notPinned | pinnedWE  ), empty, blackPieces ), whiteOrtho );
        SIMD    slidingMovesSW      = MaskOut( SlideIntoSW( whiteDiag  & (notPinned | pinnedSWNE), empty, blackPieces ), whiteDiag  );
        SIMD    slidingMovesS       = MaskOut( SlideIntoS(  whiteOrtho & (notPinned | pinnedNS  ), empty, blackPieces ), whiteOrtho );
        SIMD    slidingMovesSE      = MaskOut( SlideIntoSE( whiteDiag  & (notPinned | pinnedNWSE), empty, blackPieces ), whiteDiag  );
        SIMD    slidingMovesE       = MaskOut( SlideIntoE(  whiteOrtho & (notPinned | pinnedWE  ), empty, blackPieces ), whiteOrtho );
        SIMD    slidingMovesNE      = MaskOut( SlideIntoNE( whiteDiag  & (notPinned | pinnedSWNE), empty, blackPieces ), whiteDiag  );

        SIMD    epTarget            = castlingAndEP & EP_SQUARES;
        SIMD    epVictim            = Shift< SHIFT_S >( epTarget );
        SIMD    epCaptor1           = whitePawns & Shift< SHIFT_W >( epVictim );
        SIMD    epCaptor2           = whitePawns & Shift< SHIFT_E >( epVictim );

        SIMD    epDiscCheckNW       = StepNW( PropNW( whiteKing, empty | epVictim ) ) & blackDiag;
        SIMD    epDiscCheckSW       = StepSW( PropSW( whiteKing, empty | epVictim ) ) & blackDiag;
        SIMD    epDiscCheckSE       = StepSE( PropSE( whiteKing, empty | epVictim ) ) & blackDiag;
        SIMD    epDiscCheckNE       = StepNE( PropNE( whiteKing, empty | epVictim ) ) & blackDiag;
        SIMD    epDiscCheckW1       = StepW(  PropW(  whiteKing, empty | epVictim | epCaptor1 ) );
        SIMD    epDiscCheckW2       = StepW(  PropW(  whiteKing, empty | epVictim | epCaptor2 ) );
        SIMD    epDiscCheckE1       = StepE(  PropE(  whiteKing, empty | epVictim | epCaptor1 ) );
        SIMD    epDiscCheckE2       = StepE(  PropE(  whiteKing, empty | epVictim | epCaptor2 ) );
        SIMD    epDiscCheckW        = (epDiscCheckW1 | epDiscCheckW2) & blackOrtho;
        SIMD    epDiscCheckE        = (epDiscCheckE1 | epDiscCheckE2) & blackOrtho;
        SIMD    epDiscCheck         = epDiscCheckNW | epDiscCheckW | epDiscCheckSW | epDiscCheckSE | epDiscCheckE | epDiscCheckNE;
        SIMD    epValidTarget       = SelectIfZero( epDiscCheck, epTarget );

        SIMD    pawnCheckMask       = checkMask | (epValidTarget & Shift< SHIFT_N >( checkMask ));
        SIMD    pawnAttacksNW       = StepNW( whitePawns & (notPinned | pinnedNWSE), blackPieces | epValidTarget ) & pawnCheckMask;
        SIMD    pawnAttacksNE       = StepNE( whitePawns & (notPinned | pinnedSWNE), blackPieces | epValidTarget ) & pawnCheckMask;
        SIMD    pawnClearN          = StepN(  whitePawns & (notPinned | pinnedNS), empty );
        SIMD    pawnDoublesN        = StepN(  pawnClearN & RANK_3, empty ) & checkMask;
        SIMD    pawnMovesN          = pawnClearN & checkMask;

        SIMD    mobileKnights       = whiteKnights & notPinned;
        SIMD    knightTargets       = ~whitePieces & checkMask;
        SIMD    knightMovesNNW      = Shift< SHIFT_N + SHIFT_NW >( mobileKnights & (~FILE_A)           ) & knightTargets;
        SIMD    knightMovesWNW      = Shift< SHIFT_W + SHIFT_NW >( mobileKnights & (~FILE_A & ~FILE_B) ) & knightTargets;
        SIMD    knightMovesWSW      = Shift< SHIFT_W + SHIFT_SW >( mobileKnights & (~FILE_A & ~FILE_B) ) & knightTargets;
        SIMD    knightMovesSSW      = Shift< SHIFT_S + SHIFT_SW >( mobileKnights & (~FILE_A)           ) & knightTargets;
        SIMD    knightMovesSSE      = Shift< SHIFT_S + SHIFT_SE >( mobileKnights & (~FILE_H)           ) & knightTargets;
        SIMD    knightMovesESE      = Shift< SHIFT_E + SHIFT_SE >( mobileKnights & (~FILE_H & ~FILE_G) ) & knightTargets;
        SIMD    knightMovesENE      = Shift< SHIFT_E + SHIFT_NE >( mobileKnights & (~FILE_H & ~FILE_G) ) & knightTargets;
        SIMD    knightMovesNNE      = Shift< SHIFT_N + SHIFT_NE >( mobileKnights & (~FILE_H)           ) & knightTargets;

        SIMD    blackPawnsCon       = StepSW( blackPawns ) | StepSE( blackPawns ); 
        SIMD    blackKnightsCon     = StepKnights( blackKnights );
        SIMD    blackDiagCon        = SlideIntoExDiag(  blackDiag,  empty | whiteKing, blackPieces );
        SIMD    blackOrthoCon       = SlideIntoExOrtho( blackOrtho, empty | whiteKing, blackPieces );
        SIMD    blackKingCon        = StepOut( blackKing );
        SIMD    blackControl        = blackPawnsCon | blackKnightsCon | blackDiagCon | blackOrthoCon | blackKingCon;

        SIMD    castleKingBlocks    = allPieces    & (SQUARE_F1 | SQUARE_G1);                
        SIMD    castleQueenBlocks   = allPieces    & (SQUARE_B1 | SQUARE_C1 | SQUARE_D1); 
        SIMD    castleKingThreats   = blackControl & (SQUARE_E1 | SQUARE_F1 | SQUARE_G1); 
        SIMD    castleQueenThreats  = blackControl & (SQUARE_C1 | SQUARE_D1 | SQUARE_E1); 
        SIMD    castleKingUnavail   = MaskOut( (SIMD) SQUARE_H1, castlingAndEP & whiteRooks );
        SIMD    castleQueenUnavail  = MaskOut( (SIMD) SQUARE_A1, castlingAndEP & whiteRooks );
        SIMD    castleKing          = SelectIfZero( (castleKingBlocks  | castleKingThreats  | castleKingUnavail),  (SIMD) SQUARE_G1 );
        SIMD    castleQueen         = SelectIfZero( (castleQueenBlocks | castleQueenThreats | castleQueenUnavail), (SIMD) SQUARE_C1 );
        SIMD    castlingMoves       = castleKing | castleQueen;
        SIMD    kingMoves           = StepOut( whiteKing, ~whitePieces & ~blackControl );

        dest->_SlidingMovesNW       = SelectIfZero( _GameResult, slidingMovesNW );
        dest->_SlidingMovesNE       = SelectIfZero( _GameResult, slidingMovesNE );
        dest->_SlidingMovesSW       = SelectIfZero( _GameResult, slidingMovesSW );
        dest->_SlidingMovesSE       = SelectIfZero( _GameResult, slidingMovesSE );
        dest->_SlidingMovesN        = SelectIfZero( _GameResult, slidingMovesN );
        dest->_SlidingMovesW        = SelectIfZero( _GameResult, slidingMovesW );
        dest->_SlidingMovesE        = SelectIfZero( _GameResult, slidingMovesE );
        dest->_SlidingMovesS        = SelectIfZero( _GameResult, slidingMovesS );
        dest->_KnightMovesNNW       = SelectIfZero( _GameResult, knightMovesNNW );
        dest->_KnightMovesNNE       = SelectIfZero( _GameResult, knightMovesNNE );
        dest->_KnightMovesWNW       = SelectIfZero( _GameResult, knightMovesWNW );
        dest->_KnightMovesENE       = SelectIfZero( _GameResult, knightMovesENE );
        dest->_KnightMovesWSW       = SelectIfZero( _GameResult, knightMovesWSW );
        dest->_KnightMovesESE       = SelectIfZero( _GameResult, knightMovesESE );
        dest->_KnightMovesSSW       = SelectIfZero( _GameResult, knightMovesSSW );
        dest->_KnightMovesSSE       = SelectIfZero( _GameResult, knightMovesSSE );
        dest->_PawnMovesN           = SelectIfZero( _GameResult, pawnMovesN );
        dest->_PawnDoublesN         = SelectIfZero( _GameResult, pawnDoublesN );
        dest->_PawnAttacksNE        = SelectIfZero( _GameResult, pawnAttacksNE );
        dest->_PawnAttacksNW        = SelectIfZero( _GameResult, pawnAttacksNW );
        dest->_CastlingMoves        = SelectIfZero( _GameResult, castlingMoves );
        dest->_KingMoves            = SelectIfZero( _GameResult, kingMoves );
        dest->_CheckMask            = SelectIfZero( _GameResult, checkMask );
    }


    //--------------------------------------------------------------------------
    /// Flip the white/black pieces to view the board "from the other side".
    ///
    /// Note that this doesn't change the side to move, or any other game state.
    ///
    /// \param  prev    Position to flip
    
    PDECL void FlipFrom( const PositionT< SIMD >& prev )
    {
        SIMD    next_WhitePawns       = ByteSwap( prev._BlackPawns   );
        SIMD    next_WhiteKnights     = ByteSwap( prev._BlackKnights );
        SIMD    next_WhiteBishops     = ByteSwap( prev._BlackBishops );
        SIMD    next_WhiteRooks       = ByteSwap( prev._BlackRooks   );
        SIMD    next_WhiteQueens      = ByteSwap( prev._BlackQueens  );
        SIMD    next_WhiteKing        = ByteSwap( prev._BlackKing    );
        SIMD    next_BlackPawns       = ByteSwap( prev._WhitePawns   );
        SIMD    next_BlackKnights     = ByteSwap( prev._WhiteKnights );
        SIMD    next_BlackBishops     = ByteSwap( prev._WhiteBishops );
        SIMD    next_BlackRooks       = ByteSwap( prev._WhiteRooks   );
        SIMD    next_BlackQueens      = ByteSwap( prev._WhiteQueens  );
        SIMD    next_BlackKing        = ByteSwap( prev._WhiteKing    );
        SIMD    next_CastlingAndEP    = ByteSwap( prev._CastlingAndEP );
        SIMD    next_BoardFlipped     = ~prev._BoardFlipped;

        _WhitePawns     = next_WhitePawns;
        _WhiteKnights   = next_WhiteKnights;  
        _WhiteBishops   = next_WhiteBishops;  
        _WhiteRooks     = next_WhiteRooks;    
        _WhiteQueens    = next_WhiteQueens;   
        _WhiteKing      = next_WhiteKing;     
        _BlackPawns     = next_BlackPawns;    
        _BlackKnights   = next_BlackKnights;  
        _BlackBishops   = next_BlackBishops;  
        _BlackRooks     = next_BlackRooks;    
        _BlackQueens    = next_BlackQueens;   
        _BlackKing      = next_BlackKing;     
        _CastlingAndEP  = next_CastlingAndEP; 
        _BoardFlipped   = next_BoardFlipped;  
        _WhiteToMove    = prev._WhiteToMove;
        _HalfmoveClock  = prev._HalfmoveClock;
        _FullmoveNum    = prev._FullmoveNum;
        _GameResult     = prev._GameResult;
    }


    PDECL void FlipInPlace()
    { 
        this->FlipFrom( *this ); 
    }

    bool operator<( const PositionT& other ) const
    {
        return(memcmp(this, &other, sizeof(*this)) == 0);
    }
};

