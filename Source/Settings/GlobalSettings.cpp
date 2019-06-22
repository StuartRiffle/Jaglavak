// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "Generated/DefaultSettings.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

void GlobalSettings::Initialize( const vector< string >& configFiles )
{
    // Start with the embedded defaults

    this->LoadJsonValues( EmbeddedFile::DefaultSettings );

    // Overlay the normal settings file

    vector< string > filesToLoad = configFiles;
    filesToLoad.insert( filesToLoad.begin(), "Settings.json" );

    // Then any config files specified on the command line

    for( const string& filename : filesToLoad )
    {
        FILE* f = fopen( filename.c_str(), "r" );
        if( !f )
            continue;

        fseek( f, 0, SEEK_END );
        size_t size = ftell( f );
        fseek( f, 0, SEEK_SET );

        string json;
        json.resize( size );

        size_t loaded = fread( (void*) json.data(), size, 1, f );
        fclose( f );

        if( loaded == size )
            this->LoadJsonValues( json );
    }
}

void GlobalSettings::LoadJsonValues( const string& json )
{
    pt::ptree tree;

    try 
    { 
        pt::read_json( std::stringstream( json ), tree ); 

    } catch( ... ) {}

    for( auto& option : tree )
    {
        const string& name = option.first;
        pt::ptree&    info = option.second;

        int value = info.get< int >( "value" );
        this->Set( name, value );
    }
}

void GlobalSettings::PrintListForUci() const
{
    for( auto& pair : _Value )
    {
        const string& name = pair.first;
        int val = pair.second;

        cout << "option type spin name " << name << " default " << val << endl;
    }
}
