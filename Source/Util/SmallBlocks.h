// WIP
#pragma once

class SmallBlockAllocator
{
    list< HugeBuffer >  _Pages;
    size_t _PageSize;
    
    vector< addr_t* > _FreeBlocks;

    SmallBlockAllocator()
    {
        _Pages.push_back();
    }

    void* Alloc( size_t bytes )
    {
        size_t slot = ROUND_UP_POW2( bytes, _Alignment ) >> _AlignShift;
        if( slot >= _FreeBlocks.size() )
            _FreeBlocks.resize( slot + 1 );

        void** ptr = _FreeBlocks[slot];
        if( ptr )
        {
            addr_t* next = *ptr;
            _FreeBlocks[slot] = next;
            return ptr;
        }

        if( mCursor + bytes > mPages.front()._Size )
            _Pages.push_front();

        HugePage& page = _Pages.front();
        {
            ptr = (void*) (page._Addr + _Offset);
            _Offset += bytes;
            return ptr;
        }

        _Pages.push_front();


        // Cut off a slice

    }
        else
        {
        void* ptr = this->CutSlice( bytes );
        mLargeBlocks[bytes].
        }



        int slot 
}

void Free(void* ptr)
{

}

};

