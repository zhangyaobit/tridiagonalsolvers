#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <memory.h>
#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <iostream>
#include <fstream>

#include "common.h"

#include "tridiagonal_gold.h"

void crpcr(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems, int restSystemSize);
void crpcrNBC(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems, int restSystemSize);
void pcr(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems);
void cr(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems);
void crNBC(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems);
void crNBC32(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems);
//void pcrGauss(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems, int gaussian, int stride, int BPS);
void pcrGaussWrapper(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems, int gaussian);
void tridiagonalMKL(T *a, T *b, T *c, T * &d, T * &x, int system_size, int num_systems);
void multiThreadTridiagonalMKL(T *a, T *b, T *c, T * &d, T * &x, int system_size, int num_systems);

int testTridiagonal()
{
    int numSystems = 2;
    int systemSize = 1024*8;
    const unsigned int memSize = sizeof(T)*numSystems*systemSize;

    T* a = (T*) malloc(memSize);
    T* b = (T*) malloc(memSize);
    T* c = (T*) malloc(memSize);
    T* d = (T*) malloc(memSize);
    T* x1 = (T*) malloc(memSize);
    T* x2 = (T*) malloc(memSize);

    for (int i = 0; i < numSystems; i++)
    {
        testGeneration(&a[i*systemSize], &b[i*systemSize], &c[i*systemSize], &d[i*systemSize], &x1[i*systemSize], systemSize);
    }

	printf("created temp data\n");
    unsigned int timer1, timer2;

    CUT_SAFE_CALL(cutCreateTimer(&timer1));
    cutStartTimer(timer1);
    /*cr(a, b, c, d, x2, systemSize, numSystems);
	pcr(a, b, c, d, x2, systemSize, numSystems);
	crNBC(a, b, c, d, x2, systemSize, numSystems);
	crNBC32(a, b, c, d, x2, systemSize, numSystems);

	//crpcr(a, b, c, d, x2, systemSize, numSystems, 512);
	crpcr(a, b, c, d, x2, systemSize, numSystems, 256);
	crpcr(a, b, c, d, x2, systemSize, numSystems, 128);
	crpcr(a, b, c, d, x2, systemSize, numSystems, 64);
	crpcr(a, b, c, d, x2, systemSize, numSystems, 32);
	crpcr(a, b, c, d, x2, systemSize, numSystems, 16);
	crpcr(a, b, c, d, x2, systemSize, numSystems, 8);
	crpcr(a, b, c, d, x2, systemSize, numSystems, 4);
	crpcr(a, b, c, d, x2, systemSize, numSystems, 2);

	//crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 512);
	crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 256);
	crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 128);
	crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 64);
	crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 32);
	crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 16);
	crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 8);*/
	//crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 4);
	//crpcrNBC(a, b, c, d, x2, systemSize, numSystems, 2);

	int success;

    //pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 512);	
	//pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 256);	
	//pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 128);	
	pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 64);	
	//pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 32);	
	/*pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 16);	
	pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 8);		
	pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 4);	*/
	//pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 2);
	
	//pcrGaussWrapper(a, b, c, d, x2, systemSize, numSystems, 1);
	

    cutStopTimer(timer1);
    //printf("numSystems: %d, systemSize: %d, GPU memory allocation + PCI-E transfer + kernel execution time: %f ms\n", numSystems, systemSize, cutGetTimerValue(timer1));
	
    CUT_SAFE_CALL(cutCreateTimer(&timer2));
    cutStartTimer(timer2);
    serialManySystems(a,b,c,d,x1,systemSize,numSystems);
	//multiThreadManySystems(a, b, c, d, x1, systemSize, numSystems);
	//tridiagonalMKL(a, b, c, d, x1, systemSize, numSystems);
	//multiThreadTridiagonalMKL(a, b, c, d, x1, systemSize, numSystems);
	cutStopTimer(timer2);
	printf("CPU: numSystems: %d, systemSize: %d, CPU time: %f ms\n", numSystems, systemSize, cutGetTimerValue(timer2));

    writeResultToFile(x1,numSystems,systemSize,"cpu_result.txt");
    writeResultToFile(x2,numSystems,systemSize,"gpu_result.txt");

	
    success = compareManySystems(x1, x2, systemSize, numSystems, 0.001f);

    free(a);
    free(b);
    free(c);
    free(d);
    free(x1);
    free(x2);

	system("PAUSE");
    return success;
}

int main( int argc, char** argv) 
{
    CUT_DEVICE_INIT(argc, argv);
    CUDA_SAFE_CALL(cudaSetDevice(0));

    testTridiagonal();

    return EXIT_SUCCESS;
}