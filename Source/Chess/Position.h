// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

/// A snapshot of the game state

template< typename SIMD >
struct ALIGN_SIMD PositionT
{
    SIMD        mWhitePawns;        /// Bitmask of the white pawns 
    SIMD        mWhiteKnights;      /// Bitmask of the white knights    
    SIMD        mWhiteBishops;      /// Bitmask of the white bishops    
    SIMD        mWhiteRooks;        /// Bitmask of the white rooks 
    SIMD        mWhiteQueens;       /// Bitmask of the white queens    
    SIMD        mWhiteKing;         /// Bitmask of the white king 
                                      
    SIMD        mBlackPawns;        /// Bitmask of the black pawns     
    SIMD        mBlackKnights;      /// Bitmask of the black knights
    SIMD        mBlackBishops;      /// Bitmask of the black bishops    
    SIMD        mBlackRooks;        /// Bitmask of the black rooks     
    SIMD        mBlackQueens;       /// Bitmask of the black queens    
    SIMD        mBlackKing;         /// Bitmask of the black king 
                                     
    SIMD        mCastlingAndEP;     /// Bitmask of EP capture targets and castling targets
    SIMD        mBoardFlipped;      /// A mask which is ~0 when this structure is white/black flipped from the actual position
    SIMD        mWhiteToMove;       /// 1 if it's white to play, 0 if black
    SIMD        mHalfmoveClock;     /// Number of halfmoves since the last capture or pawn move
    SIMD        mFullmoveNum;       /// Starts at 1, increments after black moves
    SIMD        mResult;            /// Set when the game is over

    /// Reset the fields to the start-of-game position.
    
    PDECL void Reset()
    {
        mWhitePawns         = RANK_2;
        mWhiteKnights       = SQUARE_B1 | SQUARE_G1;
        mWhiteBishops       = SQUARE_C1 | SQUARE_F1;
        mWhiteRooks         = SQUARE_A1 | SQUARE_H1;
        mWhiteQueens        = SQUARE_D1;        
        mWhiteKing          = SQUARE_E1;

        mBlackPawns         = RANK_7;              
        mBlackKnights       = SQUARE_B8 | SQUARE_G8;
        mBlackBishops       = SQUARE_C8 | SQUARE_F8;
        mBlackRooks         = SQUARE_A8 | SQUARE_H8;
        mBlackQueens        = SQUARE_D8;
        mBlackKing          = SQUARE_E8;

        mCastlingAndEP      = SQUARE_A1 | SQUARE_H1 | SQUARE_A8 | SQUARE_H8;
        mBoardFlipped       = 0;
        mWhiteToMove        = 1;
        mHalfmoveClock      = 0;
        mFullmoveNum        = 1;
        mResult             = RESULT_UNKNOWN;
    }

    /// Duplicate member values across SIMD lanes

    template< typename SCALAR >
    PDECL void Broadcast( const PositionT< SCALAR >& src )
    {
        mWhitePawns         = src.mWhitePawns;   
        mWhiteKnights       = src.mWhiteKnights; 
        mWhiteBishops       = src.mWhiteBishops; 
        mWhiteRooks         = src.mWhiteRooks;   
        mWhiteQueens        = src.mWhiteQueens;  
        mWhiteKing          = src.mWhiteKing;
                            
        mBlackPawns         = src.mBlackPawns;   
        mBlackKnights       = src.mBlackKnights; 
        mBlackBishops       = src.mBlackBishops; 
        mBlackRooks         = src.mBlackRooks;   
        mBlackQueens        = src.mBlackQueens;  
        mBlackKing          = src.mBlackKing;

        mCastlingAndEP      = src.mCastlingAndEP;
        mBoardFlipped       = src.mBoardFlipped; 
        mWhiteToMove        = src.mWhiteToMove;  
        mHalfmoveClock      = src.mHalfmoveClock;
        mFullmoveNum        = src.mFullmoveNum;  
        mResult             = src.mResult;
    }


    /// Update the game state by applying a (valid) move
    
    PDECL void Step( const MoveSpecT< SIMD >& move, MoveMapT< SIMD >* nextMapOut = NULL )
    {
        SIMD    moveSrc     = SelectWithMask( mBoardFlipped,  FlipSquareIndex( move.mSrc ), move.mSrc );
        SIMD    srcBit      = SquareBit( moveSrc );
        SIMD    isPawnMove  = SelectIfNotZero( srcBit & mWhitePawns, MaskAllSet< SIMD >() );
        SIMD    isCapture   = CmpEqual( move.mType, (SIMD) CAPTURE_LOSING )
                            | CmpEqual( move.mType, (SIMD) CAPTURE_EQUAL )
                            | CmpEqual( move.mType, (SIMD) CAPTURE_WINNING )
                            | CmpEqual( move.mType, (SIMD) CAPTURE_PROMOTE_KNIGHT )
                            | CmpEqual( move.mType, (SIMD) CAPTURE_PROMOTE_BISHOP )
                            | CmpEqual( move.mType, (SIMD) CAPTURE_PROMOTE_ROOK )
                            | CmpEqual( move.mType, (SIMD) CAPTURE_PROMOTE_QUEEN );

        this->ApplyMove( move.mSrc, move.mDest, move.mType );
        this->FlipInPlace();

        mWhiteToMove        = SelectIfNotZero( mResult, mWhiteToMove,    mWhiteToMove ^ 1 );
        mFullmoveNum        = SelectIfNotZero( mResult, mFullmoveNum,    mFullmoveNum + mWhiteToMove );
        mHalfmoveClock      = SelectIfNotZero( mResult, mHalfmoveClock, (mHalfmoveClock + 1) & ~(isPawnMove | isCapture) );

        MoveMapT< SIMD > localMap;
        MoveMapT< SIMD >* mmap = nextMapOut? nextMapOut : &localMap;

        this->CalcMoveMap( mmap );

        SIMD    moveTargets = mmap->CalcMoveTargets();
        SIMD    inCheck     = mmap->IsInCheck();
        SIMD    win         = SelectIfNotZero( mWhiteToMove, (SIMD) RESULT_BLACK_WIN, (SIMD) RESULT_WHITE_WIN );
        SIMD    winOrDraw   = SelectIfNotZero( inCheck, win, (SIMD) RESULT_DRAW );
        SIMD    gameResult  = SelectIfZero( moveTargets, winOrDraw, (SIMD) RESULT_UNKNOWN );

        mResult             = SelectIfNotZero( mResult, mResult, gameResult );
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
        SIMD    whitePawns          = mWhitePawns;    
        SIMD    whiteKnights        = mWhiteKnights;  
        SIMD    whiteBishops        = mWhiteBishops;  
        SIMD    whiteRooks          = mWhiteRooks;    
        SIMD    whiteQueens         = mWhiteQueens;   
        SIMD    whiteKing           = mWhiteKing;     
        SIMD    blackPawns          = mBlackPawns;    
        SIMD    blackKnights        = mBlackKnights;  
        SIMD    blackBishops        = mBlackBishops;  
        SIMD    blackRooks          = mBlackRooks;    
        SIMD    blackQueens         = mBlackQueens;   
        SIMD    blackKing           = mBlackKing;
        SIMD    castlingAndEP       = mCastlingAndEP;

        SIMD    moveSrc             = SelectWithMask( mBoardFlipped, FlipSquareIndex( srcIdx ),  srcIdx  );
        SIMD    moveDest            = SelectWithMask( mBoardFlipped, FlipSquareIndex( destIdx ), destIdx );
        SIMD    srcBit              = SquareBit( moveSrc );
        SIMD    destBit             = SquareBit( moveDest );

        SIMD    srcPawn             = srcBit & whitePawns;
        SIMD    srcKnight           = srcBit & whiteKnights;
        SIMD    srcBishop           = srcBit & whiteBishops;
        SIMD    srcRook             = srcBit & whiteRooks;
        SIMD    srcQueen            = srcBit & whiteQueens;
        SIMD    srcKing             = srcBit & whiteKing;

        SIMD    promotedKnight      = destBit & (CmpEqual( moveType, (SIMD) PROMOTE_KNIGHT ) | CmpEqual( moveType, (SIMD) CAPTURE_PROMOTE_KNIGHT ));
        SIMD    promotedBishop      = destBit & (CmpEqual( moveType, (SIMD) PROMOTE_BISHOP ) | CmpEqual( moveType, (SIMD) CAPTURE_PROMOTE_BISHOP ));
        SIMD    promotedRook        = destBit & (CmpEqual( moveType, (SIMD) PROMOTE_ROOK   ) | CmpEqual( moveType, (SIMD) CAPTURE_PROMOTE_ROOK   ));
        SIMD    promotedQueen       = destBit & (CmpEqual( moveType, (SIMD) PROMOTE_QUEEN  ) | CmpEqual( moveType, (SIMD) CAPTURE_PROMOTE_QUEEN  ));

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

        mWhitePawns                 = SelectIfNotZero( mResult, mWhitePawns,    whitePawns    );
        mWhiteKnights               = SelectIfNotZero( mResult, mWhiteKnights,  whiteKnights  );
        mWhiteBishops               = SelectIfNotZero( mResult, mWhiteBishops,  whiteBishops  );
        mWhiteRooks                 = SelectIfNotZero( mResult, mWhiteRooks,    whiteRooks    );
        mWhiteQueens                = SelectIfNotZero( mResult, mWhiteQueens,   whiteQueens   );
        mWhiteKing                  = SelectIfNotZero( mResult, mWhiteKing,     whiteKing     );
        mBlackPawns                 = SelectIfNotZero( mResult, mBlackPawns,    blackPawns    );
        mBlackKnights               = SelectIfNotZero( mResult, mBlackKnights,  blackKnights  );
        mBlackBishops               = SelectIfNotZero( mResult, mBlackBishops,  blackBishops  );
        mBlackRooks                 = SelectIfNotZero( mResult, mBlackRooks,    blackRooks    );
        mBlackQueens                = SelectIfNotZero( mResult, mBlackQueens,   blackQueens   );
        mCastlingAndEP              = SelectIfNotZero( mResult, mCastlingAndEP, castlingAndEP );
    }


    /// Generate a map of valid moves from the current position
    
    PDECL void CalcMoveMap( MoveMapT< SIMD >* RESTRICT dest ) const
    {
        SIMD    whitePawns          = mWhitePawns;    
        SIMD    whiteKnights        = mWhiteKnights;  
        SIMD    whiteBishops        = mWhiteBishops;  
        SIMD    whiteRooks          = mWhiteRooks;    
        SIMD    whiteQueens         = mWhiteQueens;   
        SIMD    whiteKing           = mWhiteKing;     
        SIMD    blackPawns          = mBlackPawns;    
        SIMD    blackKnights        = mBlackKnights;  
        SIMD    blackBishops        = mBlackBishops;  
        SIMD    blackRooks          = mBlackRooks;    
        SIMD    blackQueens         = mBlackQueens;   
        SIMD    blackKing           = mBlackKing;     
        SIMD    castlingAndEP       = mCastlingAndEP;

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

        dest->mSlidingMovesNW       = SelectIfZero( mResult, slidingMovesNW );
        dest->mSlidingMovesNE       = SelectIfZero( mResult, slidingMovesNE );
        dest->mSlidingMovesSW       = SelectIfZero( mResult, slidingMovesSW );
        dest->mSlidingMovesSE       = SelectIfZero( mResult, slidingMovesSE );
        dest->mSlidingMovesN        = SelectIfZero( mResult, slidingMovesN );
        dest->mSlidingMovesW        = SelectIfZero( mResult, slidingMovesW );
        dest->mSlidingMovesE        = SelectIfZero( mResult, slidingMovesE );
        dest->mSlidingMovesS        = SelectIfZero( mResult, slidingMovesS );
        dest->mKnightMovesNNW       = SelectIfZero( mResult, knightMovesNNW );
        dest->mKnightMovesNNE       = SelectIfZero( mResult, knightMovesNNE );
        dest->mKnightMovesWNW       = SelectIfZero( mResult, knightMovesWNW );
        dest->mKnightMovesENE       = SelectIfZero( mResult, knightMovesENE );
        dest->mKnightMovesWSW       = SelectIfZero( mResult, knightMovesWSW );
        dest->mKnightMovesESE       = SelectIfZero( mResult, knightMovesESE );
        dest->mKnightMovesSSW       = SelectIfZero( mResult, knightMovesSSW );
        dest->mKnightMovesSSE       = SelectIfZero( mResult, knightMovesSSE );
        dest->mPawnMovesN           = SelectIfZero( mResult, pawnMovesN );
        dest->mPawnDoublesN         = SelectIfZero( mResult, pawnDoublesN );
        dest->mPawnAttacksNE        = SelectIfZero( mResult, pawnAttacksNE );
        dest->mPawnAttacksNW        = SelectIfZero( mResult, pawnAttacksNW );
        dest->mCastlingMoves        = SelectIfZero( mResult, castlingMoves );
        dest->mKingMoves            = SelectIfZero( mResult, kingMoves );
        dest->mCheckMask            = SelectIfZero( mResult, checkMask );
    }


    //--------------------------------------------------------------------------
    /// Flip the white/black pieces to view the board "from the other side".
    ///
    /// Note that this doesn't change the side to move, or any other game state.
    ///
    /// \param  prev    Position to flip
    
    PDECL void FlipFrom( const PositionT< SIMD >& prev )
    {
        SIMD    newWhitePawns       = ByteSwap( prev.mBlackPawns   );
        SIMD    newWhiteKnights     = ByteSwap( prev.mBlackKnights );
        SIMD    newWhiteBishops     = ByteSwap( prev.mBlackBishops );
        SIMD    newWhiteRooks       = ByteSwap( prev.mBlackRooks   );
        SIMD    newWhiteQueens      = ByteSwap( prev.mBlackQueens  );
        SIMD    newWhiteKing        = ByteSwap( prev.mBlackKing    );
        SIMD    newBlackPawns       = ByteSwap( prev.mWhitePawns   );
        SIMD    newBlackKnights     = ByteSwap( prev.mWhiteKnights );
        SIMD    newBlackBishops     = ByteSwap( prev.mWhiteBishops );
        SIMD    newBlackRooks       = ByteSwap( prev.mWhiteRooks   );
        SIMD    newBlackQueens      = ByteSwap( prev.mWhiteQueens  );
        SIMD    newBlackKing        = ByteSwap( prev.mWhiteKing    );
        SIMD    newCastlingAndEP    = ByteSwap( prev.mCastlingAndEP );
        SIMD    newBoardFlipped     = ~prev.mBoardFlipped;

        mWhitePawns                 = newWhitePawns;
        mWhiteKnights               = newWhiteKnights;  
        mWhiteBishops               = newWhiteBishops;  
        mWhiteRooks                 = newWhiteRooks;    
        mWhiteQueens                = newWhiteQueens;   
        mWhiteKing                  = newWhiteKing;     
        mBlackPawns                 = newBlackPawns;    
        mBlackKnights               = newBlackKnights;  
        mBlackBishops               = newBlackBishops;  
        mBlackRooks                 = newBlackRooks;    
        mBlackQueens                = newBlackQueens;   
        mBlackKing                  = newBlackKing;     
        mCastlingAndEP              = newCastlingAndEP; 
        mBoardFlipped               = newBoardFlipped;  
        mWhiteToMove                = prev.mWhiteToMove;
        mHalfmoveClock              = prev.mHalfmoveClock;
        mFullmoveNum                = prev.mFullmoveNum;
        mResult                     = prev.mResult;
    }


    PDECL void FlipInPlace()
    { 
        this->FlipFrom( *this ); 
    }
};

