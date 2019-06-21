// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
// GENERATED CODE - DO NOT EDIT THIS

namespace EmbeddedFile { 
const char* DefaultSettings = R"EMBEDDED_FILE(
{
    "EnableMulticore":         { "value": 1,    "desc": "Use all available CPU cores" },   
    "EnableSimd":              { "value": 1,    "desc": "Detect and use SIMD instructions on CPU" },
    "ForceSimdLevel":          { "value": 0,    "desc": "Override detected SIMD level [1, 2, 4, 8]" },
    "CpuDispatchThreads":      { "value": 2,    "desc": "" },
    "CpuSearchFibers":         { "value": 64,   "desc": "" },       
    "CpuAffinityMask":         { "value": 0,    "desc": "" },      
    "CpuBatchSize":            { "value": 128,  "desc": "" },      
    "EnableCuda":              { "value": 0,    "desc": "" },      
    "CudaHeapMegs":            { "value": 64,   "desc": "" },     
    "CudaAffinityMask":        { "value": 1,    "desc": "" },      
    "CudaBatchSize":           { "value": 8192, "desc": "" },       
    "MaxTreeNodes":            { "value": 1000, "desc": "" },   
    "ExplorationFactor":       { "value": 141,  "desc": "" },
    "BranchesToExpandAtLeaf":  { "value": 0,    "desc": "A value of zero means to expand them all." },
    "NumPlayoutsAtLeaf":       { "value": 8,    "desc": "" },  
    "MaxPlayoutMoves":         { "value": 200,  "desc": "" },        
    "DrawsWorthHalf":          { "value": 1,    "desc": "" },      
    "FixedRandomSeed":         { "value": 0,    "desc": "" },      
    "FlushEveryBatch":         { "value": 1,    "desc": "" },      
    "TimeSafetyBuffer":        { "value": 100,  "desc": "" },        
    "UciUpdateDelay":          { "value": 500,  "desc": "" }
}
)EMBEDDED_FILE"; 
};
