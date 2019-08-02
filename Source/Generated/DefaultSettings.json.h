// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
// GENERATED CODE - DO NOT EDIT THIS

const char* Embedded_DefaultSettings = R"EMBEDDED_FILE(
{
    "CPU.Enabled":                  1,       
    "CPU.LimitCores":               -2,      
    "CPU.LimitPlayoutCores":        1,      
    "CPU.DispatchThreads":          1,
    "CPU.SearchFibers":             8192,    
    "CPU.SIMD.Enabled":             0,
    "CPU.SIMD.ForceLevel":          0,

    "CUDA.Enabled":                 1,       
    "CUDA.HeapMegs":                512,     
    "CUDA.AffinityMask":            1,       
    "CUDA.BatchSize":               4096,     

    "Search.NumPlayoutsEach":       10,       
    "Search.BranchesToExpand":      0,       
    "Search.BatchSize":             128,      
    "Search.MaxPlayoutMoves":       500,     
    "Search.DrawsWorthHalf":        1,       
    "Search.ExplorationFactor":     1.41,
    "Search.VirtualLoss":           0.01,
    "Search.PriorNoise":            0.01,
    "Search.MaxTreeNodes":          10000000,

    "UCI.TimeSafetyMargin":         100,     
    "UCI.UpdateTime":               5000,    

    "Client.RpcThreads":            1,

    "Debug.ForceRandomSeed":        0,       
    "Debug.FlushEveryBatch":        0,       
    "Debug.DoPlayoutsInline":       0       
}


)EMBEDDED_FILE";
