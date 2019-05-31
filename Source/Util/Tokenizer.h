// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class Tokenizer
{
    vector< char > _Str;
    char*          _Cursor;

    void SkipWhite()    { while( *_Cursor &&  isspace( *_Cursor ) ) _Cursor++; }
    void SkipNonWhite() { while( *_Cursor && !isspace( *_Cursor ) ) _Cursor++; }

public:

    Tokenizer( const char* str ) 
    {
        this->Set( str );
    }

    void Set( const char* str )
    {
        size_t len = strlen( str );
        _Str.clear();
        _Str.reserve( len + 1 );
        _Str.insert( _Str.end(), str, str + len + 1 );

        _Cursor = &_Str[0];
        this->SkipWhite();
    }

    bool StartsWith( const char* target )
    {
        size_t targetLen = strlen( target );
        return( strnicmp( _Cursor, target, targetLen ) == 0 );
    }

    bool Consume( const char* target )
    {
        size_t targetLen = strlen( target );
        if( strnicmp( _Cursor, target, targetLen ) != 0 )
            return( false );

        if( _Cursor[targetLen] && !isspace( _Cursor[targetLen] ) )
            return( false );

        _Cursor += targetLen;
        this->SkipWhite();

        return( true );
    }

    const char* ConsumeNext()
    {
        const char* start = _Cursor;

        this->SkipNonWhite();
        this->SkipWhite();

        if( _Cursor > start )
        {
            if( *_Cursor )
                _Cursor[-1] = '\0';

            return( start );
        }

        return( NULL );
    }

    bool ConsumePosition( Position& pos )
    {
        const char* after = StringToPosition( _Cursor, pos );
        if( after == NULL )
            return( false );

        _Cursor = (char*) after;
        this->SkipWhite();

        return( true );
    }

    const char* ConsumeAll()
    {
        const char* start = _Cursor;

        while( *_Cursor )
            _Cursor++;

        while( (_Cursor > start) && isspace( _Cursor[-1] ) )
            _Cursor--;

        *_Cursor = '\0';
        return( start );
    }

    u64 ConsumeInt64()
    {
        const char* numstr  = this->ConsumeNext();
        u64         value   = 0;

        if( numstr )
            while( isdigit( *numstr ) )
                value = (value * 10) + (*numstr++ - '0');

        return( value );
    }

    int ConsumeInt()
    {
        return( (int) this->ConsumeInt64() );
    }

	float ConsumeFloat()
	{
		const char* numstr  = this->ConsumeNext();

		if( numstr )
			return( (float) atof( numstr ) );

		return( 0.0f );
	}
};

