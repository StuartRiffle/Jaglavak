// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "UciInterface.h"
#include "TreeSearch/IGameState.h"

#include "boost/program_options.hpp"

using namespace boost;
namespace po = program_options;

int main( int argc, char** argv )
{
    setvbuf( stdout, NULL, _IONBF, 0 );

    cout << "Jaglavak Chess " <<
        VERSION_MAJOR << "." <<
        VERSION_MINOR << "." <<
        VERSION_PATCH << endl;

    po::variables_map variables;
    po::options_description options( "Options" );
    options.add_options()
        ("config,c",    po::value< vector< string > >(), "load JSON configuration file")
        ("test,t",      "run integrated unit tests")
        ("uci,u",       po::value< vector< string > >(), "run UCI command after startup")
        ("version,v",   "print the program version and exit")
        ("help,h",      "show this message");

    try
    {
        po::store( po::parse_command_line( argc, argv, options ), variables );
        po::notify( variables );
    }
    catch( ... )
    {
        cout << "ERROR" << endl;
        cout << options;
        return -1;
    }

    if( variables.count( "version" ) )
    {
        return 0;
    }

    if( variables.count( "help" ) )
    {
        cout << options;
        return 0;
    }

    if( variables.count( "test" ) )
    {
        cout << "augh" << endl;

        extern int RunUnitTests( const char* argv0 );
        int testResult = RunUnitTests( argv[0] );

        cout << "Unit tests returned " << testResult << endl;
        return testResult;
    }

    vector< string > configFiles;
    if( variables.count( "config" ) )
        configFiles = variables["config"].as< vector< string > >();

    GlobalSettings settings;
    settings.Initialize( configFiles );

    UciInterface engine( &settings );

    if( variables.count( "uci" ) )
        for( auto& cmd : variables["uci"].as< vector< string > >() )
            engine.ProcessCommand( cmd.c_str() );

#if 1//DEBUG
    //engine.ProcessCommand( "uci" );
//    engine.ProcessCommand( "position startpos moves d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7 d1c2 f8d6 g2g4 d6b4 c1d2 d8e7 h1g1 b4c3 d2c3 f6e4 e1c1 e8g8 c3e1 f7f5 g4g5 f8e8 g5g6 h7g6 c4d5 c6d5 h2h4 e7f6 h4h5 g6h5 c2a4 f5f4 a4d7 c8d7 f1e2 f6h6 g1f1 f4e3" );
//    engine.ProcessCommand( "position startpos fen r3qb1k/1b4p1/p2pr2p/3n4/Pnp1N1N1/6RP/1B3PP1/1B1QR1K1 w - -" );
    engine.ProcessCommand( "position startpos moves e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1g5 b8d7 f2f4 e7e6 d1e2 f8e7 e1c1 d8c7 g2g4 b7b5 a2a3 c8b7" );
    engine.ProcessCommand( "go movetime 300000" );
#endif

    //PlatSetThreadName( "_YOURMOM" );

    string cmd;
    while( getline( cin, cmd ) ) 
        if( !engine.ProcessCommand( cmd.c_str() ) )
            break;

#if TOOLCHAIN_MSVC
    TerminateProcess( GetCurrentProcess(), 0 );
#endif

    return 0;
}
