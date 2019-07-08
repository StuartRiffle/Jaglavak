#pragma once
// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "Util/FEN.h"

struct GameRecordParam
{
    string _Key;
    string _Value;
}

struct GameRecord;
{
    map< string, string > _Params;
    vector< string > _Moves;
    string _FenStart;
    string _FenFinal;
    int _Result;
}

// ImportEpd.py converts PGN files to JSON

static vector< GameRecord > LoadGameRecordsFromJson( const char* json )
{
    vector< GameRecord > result;

    pt::ptree tree;

    try 
    { 
        pt::read_json( stringstream( json ), tree ); 

    } catch( ... ) {}

    for( auto& game : tree )
    {
        const string& name = option.first;
        pt::ptree&    info = option.second;

        GameRecord gameRec;
        gameRec._Moves = game.get< string >( "Moves" );
        gameRec._FenStart = game.get< string >( "FenFinal" );
        gameRec._FenFinal = game.get< string >( "FenFinal" );

    }


    for( auto& option : tree )
    {
        int value = info.get< int >( "value" );
    }
      
    return result;
}






    vector< vector< string > > epdLines = SplitLinesIntoFields( epdFile, "\r\n", ";", "#" );
    for( auto& epdLine : epdLines )
    {
        if( epdLine.size() < 1 )
            continue;

        string fen = fieldList[0];

        Position pos;
        bool fenValid = (StringToPosition( fen.c_str(), pos ) != NULL);
        if( !fenValid )
            continue;

        EpdLine line;
        line._Fen = fen;



        FenAndParams fap = { fen, {} };
        _Elem.push_back( fap );

        MapParamToValue& params = _Elem.back().second;
        for( int i = 1; i < epdLine.size(); i++ )
        {
            string& field = epdLine[i];

            size_t spacePos = field.find_first( ' ' );
            if( (spacePos > 0) && (spacePos != string::npos) )
            {
                string key   = field.substr( 0, spacePos );
                string value = field.substr( spacePos + 1 );

                params[key] = value;
            }


            params[fieldKey] = restOfField;

            REQUIRE( field[0] == 'D' );

            size_t after = 0;
            int depth = stoi( field.substr( 1 ), &after );

            REQUIRE( depth > 0 );
            REQUIRE( after > 0 );
            REQUIRE( after != string::npos );
            REQUIRE( field[after] == ' ' );
            after++;

            u64 perftTarget = stoull( field.substr( after ) );
            REQUIRE( perftTarget > 0 );

            REQUIRE( depthToTarget.find( depth ) == depthToTarget.end() );
            depthToTarget[depth] = perftTarget;
        }
            }

            targetsByPosition[pos] = move( depthToTarget );
        }

        THEN( "there is at least one solution given for each position" )
            REQUIRE( fieldList.size() > 1 );
    }
    }
    }

    WHEN( "the move generator calculates Perft" )
    {
        for( auto iter : targetsByPosition )
        {
            Position pos = iter.first;
            string fen = SerializePosition( pos );

            map< int, u64 >& depthToTarget = iter.second;
            for( auto& elem : depthToTarget )
            {
                int depth = elem.first;
                assert( depth > 0 );

                u64 target = elem.second;
                assert( target > 0 );

                THEN( "Perft of (" << fen << ") at depth " << depth << " is " << target )
                {
                    u64 calculated = CalcPerft( pos, depth );
                    REQUIRE( calculated == target );
                }
            }
        }
    }
}
}

