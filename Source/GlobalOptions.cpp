// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "GlobalOptions.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

#define OPTION_INDEX( _FIELD ) (offsetof( GlobalOptions, _##_FIELD ) / sizeof( int )), #_FIELD

static OptionInfo[] sOptionIndex =
{
    OPTION_INDEX( EnableMulticore ),        
    OPTION_INDEX( EnableSimd ),             
    OPTION_INDEX( ForceSimdLevel ),         
    OPTION_INDEX( CpuWorkThreads ),         
    OPTION_INDEX( CpuSearchFibers ),        
    OPTION_INDEX( CpuAffinityMask ),        
    OPTION_INDEX( CpuBatchSize ),           
    OPTION_INDEX( EnableCuda ),             
    OPTION_INDEX( CudaHeapMegs ),           
    OPTION_INDEX( CudaAffinityMask ),       
    OPTION_INDEX( CudaBatchSize ),          
    OPTION_INDEX( MaxTreeNodes ),           
    OPTION_INDEX( NumPlayouts ),            
    OPTION_INDEX( MaxPlayoutMoves ),        
    OPTION_INDEX( DrawsWorthHalf ),         
    OPTION_INDEX( ExplorationFactor ),      
    OPTION_INDEX( BranchesToExpandAtLeaf ), 
    OPTION_INDEX( FlushEveryBatch ),        
    OPTION_INDEX( FixedRandomSeed ),        
    OPTION_INDEX( SearchSleepTime ),        
    OPTION_INDEX( TimeSafetyBuffer ),       
    OPTION_INDEX( UciUpdateDelay ),         
};

void GlobalOptions::LoadJsonValues( string json )
{
    pt::ptree tree;
    std::stringstream ss;

    ss << json;
    pt::read_json( ss, tree );

    for( auto& option : tree )
    {
        string name = option.first;

        property_tree& info = option.second;
        int value = info.get< int >( "value" );

        this->SetOptionByName( name.c_str(), value );
    }
}

void GlobalOptions::Initialize( vector< string >& configFiles )
{
    // Initialize with the embedded defaults

    this->LoadJsonValues( Embedded::DefaultSettings_json );

    // Apply the config files in order, each overriding the last

    for( string& filename : filesToLoad )
    {
        FILE* f = fopen( filename.c_str(), "r" );
        if( !f )
            continue;

        fseek( f, 0, SEEK_END );
        size_t size = ftell( f );
        fseek( f, 0, SEEK_SET );

        string json;
        json.resize( size );

        size_t loaded = fread( contents.data(), size, 1, f );
        fclose( f );

        if( loaded == size )
            this->LoadJsonValues( json );
    }
}

bool GlobalOptions::SetOptionByName( const char* name, int value )
{
    for( int i = 0; i < NUM_ELEMENTS( sOptionIndex ); i++ )
    {
        OptionInfo& info = sOptionIndex[i];

        if( !stricmp( name, info._Name ) )
        {
            _OptionByIndex[info._Index] = value;
            return true;
        }
    }

    return false;
}

