#pragma once
struct PidConfig
{
    float  mProportionalGain;
    float  mIntegralGain;
    float  mDerivativeGain;
    float  mBias;
};

class PidController
{

    float  mVariable;
    float  mTarget;
    float  mAddError;
    float  mPrevError;
    Config  mPidConfig;

    PidController()
    {

    }

    float Update( float value, float dt )
    {
        float error = mTarget - value + mConfig.mBias;
        float slope = (mPrevError - error) / dt;

        mAddError += error * dt;
        mPrevError = error;

        mVariable =
            mConfig.mProportionalGain * error +
            mConfig.mIntegralGain     * mAddError +
            mConfig.mDerivativeGain   * slope +
            mConfig.mBias;

        return mVariable;
    }

    void SetParameters( float kp, float ki, float kd )
    {
    }
};

template< typename T >
class ManagedQueue
{
    Queue< T >      mQueue;
    PidController   mLengthController;
    int             mBufferLength;
    u64             m

    void PushBlocking( const T& elem )
    {
    }

    bool TryPush( const T& elem )
    {
    }

    bool TryPop( T& dest )
    {
    }
    // controller watches the done queue


};

