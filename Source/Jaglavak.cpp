// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "UciInterface.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

int main( int argc, char** argv )
{
    cout << "JAGLAVAK " << 
        VERSION_MAJOR << "." <<  
        VERSION_MINOR << "." << 
        VERSION_PATCH << endl;

    po::options_description options( "Allowed options" );
    options.add_options()
        ("config,C",    "load JSON configuration file")
        ("uci,U",       "run UCI command after startup")
        ("help",        "show this help message");

    po::variables_map variables;
    po::store( po::parse_command_line( argc, argv, options ), variables );
    po::notify( variables );    

    vector< string > configFiles = variables["config"].as< vector< string > >();
    configFiles.insert( configFiles.begin(), "Settings.json" );

    GlobalSettings settings;
    settings.Initialize( configFiles );

    unique_ptr< UciInterface > engine( new UciInterface( &settings ) );

    for( auto& cmd : variables["uci"].as< vector< string > >() )
        engine->ProcessCommand( cmd.c_str() );

    string cmd;
    while( getline( std::cin, cmd ) ) 
        if( !engine->ProcessCommand( cmd.c_str() ) )
            break;

    return 0;
}
