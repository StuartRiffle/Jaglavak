// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct Timer
{
    u64 mStartTime;

    Timer() { this->Reset(); }
    Timer( const Timer& rhs ) : mStartTime( rhs.mStartTime ) {}

    void Reset()         
    { 
        mStartTime = CpuInfo::GetClockTick(); 
    }

    u64 GetElapsedTicks() const
    {
        return CpuInfo::GetClockTick() - mStartTime;
    }

    float GetElapsedSec() const
    {
        return this->GetElapsedTicks() * 1.0f / CpuInfo::GetClockFrequency();
    }

    int GetElapsedMs() const
    {
        return (int) (this->GetElapsedTicks() * 1000 / CpuInfo::GetClockFrequency());
    }
};
