// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class AsyncWorker
{
public:
    virtual void Update() {}
    virtual bool Initialize() { return true; }
};

