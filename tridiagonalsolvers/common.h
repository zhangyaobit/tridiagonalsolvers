#include <cutil.h>
#include <stdio.h>
#include <cutil_inline.h>

//#define USE_DOUBLE
//#define GPU_PRINTF

#ifdef USE_DOUBLE
    #define T double
#else
    #define T float
#endif

#define numIterations 50
#define MINSYSTEM 1024

#define SMALLBLOCKSIZE 256