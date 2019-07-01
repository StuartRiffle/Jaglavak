// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"

#define CATCH_CONFIG_RUNNER  
#include "catch.hpp"

int RunUnitTests( const char* argv0 )
{
    const char* argv[]
    {
        argv0,
        "--abort",
        "--use-colour", "yes",
    };

    int argc = NUM_ELEMENTS( argv );
    int result = Catch::Session().run( argc, argv );    
        
    return result;
}

