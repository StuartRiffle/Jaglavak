// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "UciInterface.h"
#include "TreeSearch/IGameState.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

int main( int argc, char** argv )
{
    po::options_description options( "Allowed options" );
    options.add_options()
        ("config,c",    po::value< vector< string > >(), "load JSON configuration file")
        ("uci,u",       po::value< vector< string > >(), "run UCI command after startup")
        ("test,t",      "run integrated unit tests")
        ("version,v",   "print the program version and exit")
        ("help,h",      "show this message");

    cout << "Jaglavak " <<
        VERSION_MAJOR << "." <<
        VERSION_MINOR << "." <<
        VERSION_PATCH << endl;

    po::variables_map variables;
    po::store( po::parse_command_line( argc, argv, options ), variables );
    po::notify( variables );    

    if( variables.count( "version" ) )
        return(0);

    vector< string > configFiles;
    if( variables.count( "config" ) )
        configFiles = variables["config"].as< vector< string > >();

    GlobalSettings settings;
    settings.Initialize( configFiles );

    if( variables.count( "test" ) )
    {
        extern int RunUnitTests();
        int testResult = RunUnitTests();

        return testResult;
    }

    UciInterface engine( &settings );

    if( variables.count( "uci" ) )
        for( auto& cmd : variables["uci"].as< vector< string > >() )
            engine.ProcessCommand( cmd.c_str() );

    engine.ProcessCommand( "uci" );
    engine.ProcessCommand( "go" );


    string cmd;
    while( getline( std::cin, cmd ) ) 
        if( !engine.ProcessCommand( cmd.c_str() ) )
            break;

    return 0;
}
