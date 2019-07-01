// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "Player/SSE4.h"
#include "Player/AVX2.h"
#include "Player/AVX512.h"
#include "Generated/TestGames.json.h"
#include "Util/CpuInfo.h"

#incl#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;


template< typename T >
static void TestSimdType()
{
    int LANES = (int) T::LANES;
    if( LANES > CpuInfo::GetSimdLevel() )
        return;

    EpdFile epdFile;
    extern const char* Embedded_TestGames;
    LoadEpdFile( &epdFile, Embedded_TestGames );

    int count = epdFile.GetNumLines();
    int simdCount = (count + LANES - 1) & ~(LANES - 1);
    int ignored = count - (simdCount * LANES);

    pt::ptree tree;

    try
    {
        pt::read_json( std::stringstream( json ), tree );

    }
    catch( ... ) {}

    for( auto& option : tree )
    {
        const string& name = option.first;
        pt::ptree& info = option.second;

        int value = info.get< int >( "value" );
        this->Set( name, value );
    }

    Position ALIGN_SIMD startPos[LANES] = { 0 };
    Position ALIGN_SIMD finalPos[LANES] = { 0 };

    PositionT< T > ALIGN_SIMD startPosSimd;

}

TEST_CASE( "SIMD game state update" )
{

}

