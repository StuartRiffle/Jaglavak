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

    // Apply the config files in order, each potentially overridden
    // by the ones that follow

    for( const string& filename : configFiles )
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
    std::stringstream ss;

    ss << json;
    pt::read_json( ss, tree );

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

