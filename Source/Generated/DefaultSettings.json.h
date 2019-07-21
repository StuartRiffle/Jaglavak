// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
// GENERATED CODE - DO NOT EDIT THIS

const char* Embedded_DefaultSettings = R"EMBEDDED_FILE(
{
    "CPU.Enabled":                  0,       
    "CPU.LimitCores":               -2,      
    "CPU.LimitPlayoutCores":        12,      
    "CPU.DispatchThreads":          4,
    "CPU.SearchFibers":             8192,    
    "CPU.SIMD.Enabled":             0,
    "CPU.SIMD.ForceLevel":          0,

    "CUDA.Enabled":                 1,       
    "CUDA.HeapMegs":                512,     
    "CUDA.AffinityMask":            1,       
    "CUDA.BatchSize":               128,     

    "Search.NumPlayoutsEach":       1,       
    "Search.BranchesToExpand":      1,       
    "Search.BatchSize":             16,      
    "Search.MaxPlayoutMoves":       200,     
    "Search.DrawsWorthHalf":        1,       
    "Search.ExplorationFactor":     1.41,
    "Search.VirtualLoss":           0.1,     
    "Search.MaxTreeNodes":          100000,

    "UCI.TimeSafetyMargin":         100,     
    "UCI.UpdateTime":               3000,    

    "Debug.ForceRandomSeed":        0,       
    "Debug.FlushEveryBatch":        0,       
    "Debug.DoPlayoutsInline":       0       
}


)EMBEDDED_FILE";
