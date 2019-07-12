// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Jaglavak.h"

vector< vector< string > >
SplitLinesIntoFields( const char* str, const char* lineDelims, const char* fieldDelims, const char* commentDelims );
