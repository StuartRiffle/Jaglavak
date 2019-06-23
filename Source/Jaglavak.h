#pragma once

#include "Platform.h"
#include "Version.h"
#include "Chess/Core.h"

#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <unordered_map>

using std::atomic;
using std::string;    
using std::vector;
using std::list;
using std::map;
using std::set;
using std::mutex;
using std::condition_variable;
using std::thread;
using std::shared_ptr;
using std::unique_ptr;
using std::unique_lock;
using std::cout;
using std::endl;
using std::unordered_map;

#include "Settings/GlobalSettings.h"
#include "Util/Random.h"
#include "Util/Queue.h"
#include "Util/Timer.h"
#include "Util/FiberSet.h"
#include "Worker/AsyncWorker.h"

#include "Player/ScoreCard.h"
#include "Player/PlayoutParams.h"
#include "TreeSearch/PlayoutBatch.h"
