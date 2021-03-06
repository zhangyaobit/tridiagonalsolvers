#include "common.h"
#include "cr_kernel.cu"

void cr(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
{
    const unsigned int num_threads_block = systemSize/2;
    const unsigned int memSize = sizeof(T)*numSystems*systemSize;

    // allocate device memory input and output arrays
    T* d_a;
    T* d_b;
    T* d_c;
    T* d_d;
    T* d_x;

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    cutStartTimer(timer);

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x,memSize));

    cutStopTimer(timer);
    //printf("GPU cudaMalloc time: %f ms\n", cutGetTimerValue(timer));

   // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( d_a, a,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_b, b,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_c, c,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_d, d,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_x, x,memSize, cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

	cutResetTimer(timer);
    cutStartTimer(timer);
	//i feel more confident in the timings if we do multiple iterations and average them
	for(int i = 0; i < numIterations;i ++) 
		crKernel<<< grid, threads,systemSize*5*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x);

    cudaThreadSynchronize();
    cutStopTimer(timer);
	printf("cr: numSystems: %d, systemSize: %d, GPU kernel time: %f ms\n", numSystems, systemSize, cutGetTimerValue(timer)/numIterations);
	
	CUDA_SAFE_CALL( cudaMemcpy( d_a, a,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_b, b,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_c, c,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_d, d,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_x, x,memSize, cudaMemcpyHostToDevice));
    
	// copy result from device to host	    
	crKernel<<< grid, threads,systemSize*5*sizeof(T)>>>(d_a, d_b, d_c, d_d, d_x);
    CUDA_SAFE_CALL( cudaMemcpy(x, d_x,memSize, cudaMemcpyDeviceToHost));

    // cleanup memory
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    CUDA_SAFE_CALL(cudaFree(d_d));
    CUDA_SAFE_CALL(cudaFree(d_x));
}


