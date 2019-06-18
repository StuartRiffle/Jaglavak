// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <iostream>

using std::atomic;
using std::vector;
using std::list;
using std::string;    
using std::mutex;
using std::map;
using std::condition_variable;
using std::thread;
using std::shared_ptr;
using std::unique_ptr;
using std::unique_lock;
using std::cout;
using std::endl;

#include "ScoreCard.h"
#include "GlobalOptions.h"
#include "CpuInfo.h"
#include "Random.h"
#include "Timer.h"
#include "ThreadSync.h"
#include "Queue.h"
#include "PlayoutParams.h"
#include "PlayoutBatch.h"
#include "AsyncWorker.h"
#include "Fibers.h"
#include "TreeNode.h"
#include "TreeSearch.h"


