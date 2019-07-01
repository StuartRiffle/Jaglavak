// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Jaglavak.h"
#include "boost/algorithm/string.hpp"

#include <vector>
using std::vector;
vector< vector< string > > SplitLinesIntoFields( const char* str, const char* lineDelims, const char* fieldDelims, const char* commentDelims )
{
    vector< vector< string > > result;

    vector< string > lineList;
    boost::split( lineList, str, boost::is_any_of( lineDelims ), boost::token_compress_on );

    for( string line : lineList )
    {
        size_t commentOfs =  line.find_first_of( commentDelims );
        if( commentOfs != string::npos )
            line.erase( commentOfs );

        boost::trim( line );
        if( line.empty() )
            continue;

        vector< string > rawFields;
        boost::split( rawFields, line, boost::is_any_of( fieldDelims ), boost::token_compress_on );

        vector< string > fields;
        for( auto field : rawFields )
        {
            boost::trim( field );
            if( !field.empty() )
                fields.push_back( field );
        }

        if( !fields.empty() )
            result.push_back( std::move( fields ) );
    }

    return result;
}
