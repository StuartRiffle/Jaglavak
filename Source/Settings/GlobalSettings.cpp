// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess/Core.h"
#include "GlobalSettings.h"
#include "Generated/DefaultSettings.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

#define OPTION_INDEX( _FIELD ) (offsetof( GlobalSettings, _##_FIELD ) / sizeof( int )), #_FIELD

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

void GlobalSettings::Initialize( const vector< string >& configFiles )
{
    // Start with the embedded defaults

    this->LoadJsonValues( Embedded::DefaultSettings );

    // Apply the config files in order, each potentially overridden
    // by the ones that follow

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

void GlobalSettings::LoadJsonValues( string json )
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

        this->SetValueByName( name.c_str(), value );
    }
}

void GlobalSettings::PrintListForUci()
{
    for( int i = 0; i < NUM_ELEMENTS( sOptionIndex ); i++ )
    {
        OptionInfo& info = sOptionIndex[i];
        cout << "option type spin name " << info._Name << " default " info._Value << endl;
    }
}

