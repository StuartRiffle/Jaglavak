#pragma once
struct PidConfig
{
    double  _ProportionalGain;
    double  _IntegralGain;
    double  _DerivativeGain;
    double  _Bias;
};

class PidController
{
    double      _Value;
    double      _Target;
    double      _SumError;
    double      _PrevError;
    PidConfig   _Config;

    double Update( double value, double dt )
    {
        _Value = _Config._Bias;

        double error = _Target - value + _Config._Bias;
        _Value += _Config._ProportionalGain * error;

        double slope = (_PrevError - error) / dt;
        _Value += _Config._DerivativeGain * slope;

        _SumError += error * dt;
        _Value += _Config._IntegralGain * _SumError;

        _PrevError = error;
        return _Value;
    }
};

