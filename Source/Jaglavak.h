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

using namespace std;
using namespace boost;

#include "Settings/GlobalSettings.h"
#include "Util/Random.h"
#include "Util/Queue.h"
#include "Util/Timer.h"
#include "Util/FiberSet.h"
#include "Util/StringUtil.h"
#include "Worker/AsyncWorker.h"

#include "Player/ScoreCard.h"
#include "Player/PlayoutParams.h"
#include "TreeSearch/PlayoutBatch.h"
