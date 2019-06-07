// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "UciEngine.h"
#include "Version.h"

int main( int argc, char** argv )
{
    cout << "JAGLAVAK " << 
        VERSION_MAJOR << "." <<  
        VERSION_MINOR << "." << 
        VERSION_PATCH << endl;

    unique_ptr< UciEngine > engine( new UciEngine() );

    string cmd;
    while( getline( std::cin, cmd ) ) 
        if( !engine->ProcessCommand( cmd.c_str() ) )
            break;

    return 0;
}
