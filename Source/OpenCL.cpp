// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "FEN.h"
#include "GamePlayer.h"
#include <CL/cl.h>


int Foo()
{
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    return 0;
}

