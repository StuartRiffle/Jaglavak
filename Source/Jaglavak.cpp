// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "UciEngine.h"
#include "Version.h"

int main( int argc, char** argv )
{
    setvbuf( stdin,  NULL, _IONBF, 0 );
    setvbuf( stdout, NULL, _IONBF, 0 );

    printf( "JAGLAVAK CHESS ENGINE %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH );
    printf( "Stuart Riffle\n\n" );

    auto engine( unique_ptr< UciEngine >( new UciEngine() ) );

/*
r3qb1k/1b4p1/p2pr2p/3n4/Pnp1N1N1/6RP/1B3PP1/1B1QR1K1 w - - bm Nxh6; id "Nolot.1";
r4rk1/pp1n1p1p/1nqP2p1/2b1P1B1/4NQ2/1B3P2/PP2K2P/2R5 w - - bm Rxc5; id "Nolot.2";
r2qk2r/ppp1b1pp/2n1p3/3pP1n1/3P2b1/2PB1NN1/PP4PP/R1BQK2R w KQkq - bm Nxg5; id "Nolot.3";
r1b1kb1r/1p1n1ppp/p2ppn2/6BB/2qNP3/2N5/PPP2PPP/R2Q1RK1 w kq - bm Nxe6; id "Nolot.4";
r2qrb1k/1p1b2p1/p2ppn1p/8/3NP3/1BN5/PPP3QP/1K3RR1 w - - bm e5; id "Nolot.5";
rnbqk2r/1p3ppp/p7/1NpPp3/QPP1P1n1/P4N2/4KbPP/R1B2B1R b kq - bm axb5; id "Nolot.6";
1r1bk2r/2R2ppp/p3p3/1b2P2q/4QP2/4N3/1B4PP/3R2K1 w k - bm Rxd8+; id "Nolot.7";
r3rbk1/ppq2ppp/2b1pB2/8/6Q1/1P1B3P/P1P2PP1/R2R2K1 w - - bm Bxh7+; id "Nolot.8";
r4r1k/4bppb/2n1p2p/p1n1P3/1p1p1BNP/3P1NP1/qP2QPB1/2RR2K1 w - - bm Ng5; id "Nolot.9";
r1b2rk1/1p1nbppp/pq1p4/3B4/P2NP3/2N1p3/1PP3PP/R2Q1R1K w - - bm Rxf7; id "Nolot.10";
r1b3k1/p2p1nP1/2pqr1Rp/1p2p2P/2B1PnQ1/1P6/P1PP4/1K4R1 w - - bm Rxh6; id "Nolot.11";
*/
    engine->ProcessCommand( "uci" );
    //engine->ProcessCommand( "position startpos fen r3qb1k/1b4p1/p2pr2p/3n4/Pnp1N1N1/6RP/1B3PP1/1B1QR1K1 w - -" );
    engine->ProcessCommand( "position startpos moves e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3 g8e7 d1g4 d8c7 g4g7 h8g8 g7h7 c5d4 g1e2 b8c6 f2f4 d4c3 h2h4 d5d4 e2c3 d4c3 h4h5 c8d7 f4f5 c7e5 e1f2 g8h8 f1e2 h8h7 c1e3 e7f5 e3c1 e5c5 c1e3 c5e3" );
    engine->ProcessCommand( "go" );
    
    while( !feof( stdin ) )
    {
        char buf[8192];

        const char* cmd = fgets( buf, sizeof( buf ), stdin );
        if( cmd == NULL )
            continue;

        bool timeToExit = engine->ProcessCommand( cmd );
        if( timeToExit )
            break;
    }

    return 0;
}


