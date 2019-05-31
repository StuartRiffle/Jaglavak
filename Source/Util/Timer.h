// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct Timer
{
    u64 _StartTime;

    Timer() { this->Reset(); }
    Timer( const Timer& rhs ) : _StartTime( rhs._StartTime ) {}

    void Reset()         
    { 
        _StartTime = CpuInfo::GetClockTick(); 
    }

    u64 GetElapsedTicks() const
    {
        return CpuInfo::GetClockTick() - _StartTime;
    }

    float GetElapsedSec() const
    {
        static float sInvFreq = 1.0f / CpuInfo::GetClockFrequency(); 
        return this->GetElapsedTicks() * sInvFreq;
    }

    int GetElapsedMs() const
    {
        return (int) (this->GetElapsedTicks() * 1000 / CpuInfo::GetClockFrequency());
    }
};
