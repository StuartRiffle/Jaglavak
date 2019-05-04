// Tokenizer.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef JAGLAVAK_TOKENIZER_H__
#define JAGLAVAK_TOKENIZER_H__


class Tokenizer
{
    std::vector< char > mStr;
    char*               mCursor;

    void SkipWhite()    { while( *mCursor &&  isspace( *mCursor ) ) mCursor++; }
    void SkipNonWhite() { while( *mCursor && !isspace( *mCursor ) ) mCursor++; }

public:

    Tokenizer( const char* str ) 
    {
        this->Set( str );
    }

    void Set( const char* str )
    {
        size_t len = strlen( str );
        mStr.clear();
        mStr.reserve( len + 1 );
        mStr.insert( mStr.end(), str, str + len + 1 );

        mCursor = &mStr[0];
        this->SkipWhite();
    }

    bool StartsWith( const char* target )
    {
        size_t targetLen = strlen( target );
        return( strnicmp( mCursor, target, targetLen ) == 0 );
    }

    bool Consume( const char* target )
    {
        size_t targetLen = strlen( target );
        if( strnicmp( mCursor, target, targetLen ) != 0 )
            return( false );

        if( mCursor[targetLen] && !isspace( mCursor[targetLen] ) )
            return( false );

        mCursor += targetLen;
        this->SkipWhite();

        return( true );
    }

    const char* ConsumeNext()
    {
        const char* start = mCursor;

        this->SkipNonWhite();
        this->SkipWhite();

        if( mCursor > start )
        {
            if( *mCursor )
                mCursor[-1] = '\0';

            return( start );
        }

        return( NULL );
    }

    bool ConsumePosition( Position& pos )
    {
        const char* after = StringToPosition( mCursor, pos );
        if( after == NULL )
            return( false );

        mCursor = (char*) after;
        this->SkipWhite();

        return( true );
    }

    const char* ConsumeAll()
    {
        const char* start = mCursor;

        while( *mCursor )
            mCursor++;

        while( (mCursor > start) && isspace( mCursor[-1] ) )
            mCursor--;

        *mCursor = '\0';
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

#endif // JAGLAVAK_TOKENIZER_H__
