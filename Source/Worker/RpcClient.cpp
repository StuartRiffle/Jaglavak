// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace boost;
namespace pt = boost::property_tree;
namespace http = beast::http;
namespace net = asio;
using tcp = net::ip::tcp;

#include "RpcClient.h"

static string BlockingHttpRequest( const char* host, const char* port, const char* target, const char* request );

RpcClient::RpcClient( const GlobalSettings* settings, Metrics* metrics )
{
    _Settings = settings;
    _Metrics = metrics;
}

RpcClient::~RpcClient()
{
    _TimeToExit = true;
    _CallQueue.NotifyAllWaiters();

    for( auto& thread : _RpcThreads )
        thread->join();
}

bool RpcClient::Init()
{
    _TimeToExit = false;
    
    for( int i = 0; i < _Settings->Get( "Client.RpcThreads" ); i++ )
        _RpcThreads.emplace_back( new thread( [=,this] { this->___RPC_CLIENT_THREAD___( i ); } ) );

    return true;
}

void RpcClient::AddServer( string type, string ip, string port, string target )
{
    unique_lock< mutex > lock( _Mutex );

    _ServersByType[type].push_back( ServerInfo { ip, port, target } );
}

void RpcClient::Call( const CallRef& call )
{
    _CallQueue.Push( call );
}

void RpcClient::___RPC_CLIENT_THREAD___( int idx )
{
    for( ;; )
    {
        CallRef call;
        if( !_CallQueue.Pop( call ) )
            break;
        if( _TimeToExit )
            break;

        ServerInfo serverInfo = {};
        {
            unique_lock< mutex > lock( _Mutex );

            list< ServerInfo >& servers = _ServersByType[call->_ServerType];
            assert( servers.size() > 0 );

            // Rotate the server we use
            if( servers.size() > 1 )
                servers.splice( servers.end(), servers, servers.begin() );

            if( servers.size() > 0 )
                serverInfo = servers.front();
        }

        if( !serverInfo._Addr.empty() )
        {
            try 
            { 
                stringstream request;
                pt::json_parser::write_json( request, call->_Inputs );

                string reply = BlockingHttpRequest( 
                    serverInfo._Addr.c_str(), 
                    serverInfo._Port.c_str(),
                    serverInfo._Target.c_str(),
                    request.str().c_str() );
                pt::read_json( stringstream( reply ), call->_Outputs ); 

                call->_Success = true;

            } catch( ... ) {}
        }

        call->_Done = true;
    }
}

static string BlockingHttpRequest( const char* host, const char* port, const char* target, const char* request )
{
    net::io_context ioc;
    tcp::resolver resolver( ioc );
    beast::tcp_stream stream( ioc );

    auto const results = resolver.resolve( host, port );
    stream.connect( results );

    http::request<http::string_body> req{ http::verb::post, target, 11 };

    req.set( http::field::host, host );
    req.set( http::field::user_agent, BOOST_BEAST_VERSION_STRING );
    req.set(beast::http::field::content_type, "text/plain");
    req.body() = request;
    req.prepare_payload();
    http::write( stream, req );

    beast::flat_buffer buffer;
    http::response<http::dynamic_body> res;
    http::read( stream, buffer, res );

    beast::error_code ec;
    stream.socket().shutdown( tcp::socket::shutdown_both, ec );

    return beast::buffers_to_string( res.body().data() );
}
