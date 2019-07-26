// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Jaglavak.h"
#include "StringUtil.h"

#include "boost/algorithm/string.hpp"
using namespace boost;

vector< string > SplitString( const string& str, const string& delims )
{
    vector< string > fields;
    split( fields, str, is_any_of( delims ), token_compress_on );
    return fields;
}

vector< vector< string > >
SplitLinesIntoFields( const string& str, const string& lineDelims, const string& fieldDelims, const string& commentDelims )
{
    vector< vector< string > > result;

    vector< string > lineList = SplitString( str, lineDelims );
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

/*
string CleanJson( const string& dirty )
{
    string json = dirty;
    replace_all( json, "\r", "" );

    auto lines = SplitString( json, "\n" );
    for( string line : lines )
    {
        size_t commentOfs =  line.find_first_of( "#" );
        if( commentOfs != string::npos )
            line.erase( commentOfs );
    }
}

string FormatString( string fmt, ... )
{
    const int BUFSIZE = 1024;
    char str[BUFSIZE];

    va_list args;
    va_start( args, fmt.c_str() );

    string str;
    int len = snprintf( "", 0, "" );
    str.resize( len - 1 );

    vsprintf( str, fmt.c_str(), args );
    va_end( args );

    return str;

}
  */
