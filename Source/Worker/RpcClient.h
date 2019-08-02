// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct ServerInfo
{
    string _Addr;
    string _Port;
    string _Target;
};

struct RpcCall
{
    string _ServerType;
    pt::ptree _Inputs;
    pt::ptree _Outputs;
    bool _Done;
    bool _Success;

    RpcCall() : _Done( false ), _Success( false ) {}
};

typedef std::shared_ptr< RpcCall > CallRef;
typedef ThreadSafeQueue< CallRef > CallQueue;


class RpcClient
{
    const GlobalSettings*   _Settings   = NULL;
    Metrics*                _Metrics    = NULL;
    volatile bool           _TimeToExit = false;
    CallQueue               _CallQueue;
    mutex                   _Mutex;

    map< string, list< ServerInfo > > _ServersByType;
    
    list< unique_ptr< thread > > _RpcThreads;

public:

    RpcClient( const GlobalSettings* settings, Metrics* metrics );
    ~RpcClient();

    bool Init();

    void AddServer( string type, string ip, string port, string target );
    void Call( const CallRef& call );

private:

    void ___RPC_CLIENT_THREAD___( int idx );
};
