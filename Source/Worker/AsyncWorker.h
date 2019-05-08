// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#define PLAYOUT_BATCH_MAX (4096)

class AsyncWorker
{
public:
    virtual void Update() {}
};

typedef shared_ptr< AsyncWorker > AsyncWorkerRef;
