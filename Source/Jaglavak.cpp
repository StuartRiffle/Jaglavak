// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "UciInterface.h"
#include "TreeSearch/IGameState.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

int main( int argc, char** argv )
{
    setbuf( stdout, NULL );

    cout << "JAGLAVAK " << 
        VERSION_MAJOR << "." <<  
        VERSION_MINOR << "." << 
        VERSION_PATCH << endl;

    po::options_description options( "Allowed options" );
    options.add_options()
        ("config,C",    po::value< vector< string > >(), "load JSON configuration file")
        ("uci,U",       po::value< vector< string > >(), "run UCI command after startup")
        ("version,v",   "print the program version and exit")
        ("help,h",      "show this message");

    po::variables_map variables;
    po::store( po::parse_command_line( argc, argv, options ), variables );
    po::notify( variables );    

    vector< string > configFiles;
    if( variables.count( "config" ) )
        configFiles = variables["config"].as< vector< string > >();

    GlobalSettings settings;
    settings.Initialize( configFiles );

    UciInterface engine( &settings );

    if( variables.count( "uci" ) )
        for( auto& cmd : variables["uci"].as< vector< string > >() )
            engine.ProcessCommand( cmd.c_str() );

    engine.ProcessCommand( "uci" );
    engine.ProcessCommand( "position startpos" );
    //engine.ProcessCommand( "position startpos fen 5rk1/ppp3pp/8/3pQ3/3P2b1/6P1/PP1P2K1/R1BB1r2 b - - 51 53" );
    //engine.ProcessCommand( "positions startpos fen 8/6r1/8/3Rb1Np/1p2p3/pPk5/P1P3PP/1K6 w - - 49 53" );
    //engine.ProcessCommand( "go movetime 10000" );
    //engine.ProcessCommand( "go" ); 


    string cmd;
    while( getline( std::cin, cmd ) ) 
        if( !engine.ProcessCommand( cmd.c_str() ) )
            break;

    return 0;
}
