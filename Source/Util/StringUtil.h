// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Jaglavak.h"

vector< string > SplitString( const string& str, const string& delims = ", \r\n\t" );

vector< vector< string > > SplitLinesIntoFields( 
    const string& str, 
    const string& lineDelims, 
    const string& fieldDelims, 
    const string& commentDelims );

string FormatString( const string& fmt, ... );

string CleanJson( const string& json );

