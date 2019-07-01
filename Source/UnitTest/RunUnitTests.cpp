// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"

#define CATCH_CONFIG_RUNNER  
#include "catch.hpp"

int RunUnitTests()
{
    const char* catchArgs[]
    {
        "--abort", // after first failure
        "--use-colour", "yes",
    };

    int result = Catch::Session().run( NUM_ELEMENTS( catchArgs ), catchArgs );
    return result;
}

