
#ifndef EM_H
#define EM_H

#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sys/stat.h>

#ifdef __GNUC__
#include <libgen.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

#ifdef _MSC_VER
#include <Windows.h>
#include <thread>
#endif

#include <cuda_runtime.h>

#include "PerformEM.h"
#include "emkernel.h"

using namespace std;
using namespace cv;

#endif
