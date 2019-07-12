// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Jaglavak.h"
#include "StringUtil.h"

#include "boost/algorithm/string.hpp"
using namespace boost;

vector< string > SplitString( const char* str, const char* delims = " " )
{
    vector< string > fields;
    split( fields, str, is_any_of( delims ), token_compress_on );
    return fields;
}

vector< vector< string > >
SplitLinesIntoFields( const char* str, const char* lineDelims, const char* fieldDelims, const char* commentDelims )
{
    vector< vector< string > > result;

    vector< string > lineList;
    split( lineList, str, is_any_of( lineDelims ), token_compress_on );

    for( string line : lineList )
    {
        size_t commentOfs =  line.find_first_of( commentDelims );
        if( commentOfs != string::npos )
            line.erase( commentOfs );

        trim( line );
        if( line.empty() )
            continue;

        vector< string > rawFields = SplitString( line.c_str(), fieldDelims );

        vector< string > fields;
        for( auto field : rawFields )
        {
            trim( field );
            if( !field.empty() )
                fields.push_back( field );
        }

        if( !fields.empty() )
            result.push_back( std::move( fields ) );
    }

    return result;
}
