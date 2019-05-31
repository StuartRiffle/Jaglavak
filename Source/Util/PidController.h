#pragma once
struct PidConfig
{
    float  _ProportionalGain;
    float  _IntegralGain;
    float  _DerivativeGain;
    float  _Bias;
};

class PidController
{

    float  _Variable;
    float  _Target;
    float  _AddError;
    float  _PrevError;
    Config _PidConfig;

    PidController()
    {

    }

    float Update( float value, float dt )
    {
        float error = _Target - value + _Config._Bias;
        float slope = (_PrevError - error) / dt;

        _AddError += error * dt;
        _PrevError = error;

        _Variable =
            _Config._ProportionalGain * error +
            _Config._IntegralGain     * _AddError +
            _Config._DerivativeGain   * slope +
            _Config._Bias;

        return _Variable;
    }

    void SetParameters( float kp, float ki, float kd )
    {
    }
};

