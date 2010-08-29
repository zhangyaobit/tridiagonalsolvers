#ifndef _CYCLIC_SMALL_SYSTEMS_NBC32_
#define _CYCLIC_SMALL_SYSTEMS_NBC32_

#include <stdio.h>
#include "common.h"

// at32: address translater
__device__ int at32(int x)
{
    int n = x >> 5;
    int m = x & 31;
    //printf("n = %d\n", n);
    //printf("m = %d\n", m);
    int y = n * 32 + m;
    //printf("y = %d\n", y);
    return y;
}

// cyclic reduction
__global__ void crNBC32Kernel(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;	
    int blid = blockIdx.x;
    int stride = 1;
    int thid_num = blockDim.x;
    const unsigned int systemSize = blockDim.x * 2;
    int Iteration = (int)log2(float(systemSize/2));
    
    int systemSizeExt = systemSize * 33 / 32;
    extern __shared__ char shared[];
     T* a = (T*)shared;
    T* b = (T*)&a[systemSizeExt];
    T* c = (T*)&b[systemSizeExt];
    T* d = (T*)&c[systemSizeExt];
    T* x = (T*)&d[systemSizeExt];

    a[at32(thid)] = a_d[thid + blid * systemSize];
    a[at32(thid + blockDim.x)] = a_d[thid + blockDim.x + blid * systemSize];

    b[at32(thid)] = b_d[thid + blid * systemSize];
    b[at32(thid + blockDim.x)] = b_d[thid + blockDim.x + blid * systemSize];

    c[at32(thid)] = c_d[thid + blid * systemSize];
    c[at32(thid + blockDim.x)] = c_d[thid + blockDim.x + blid * systemSize];

    d[at32(thid)] = d_d[thid + blid * systemSize];
    d[at32(thid + blockDim.x)] = d_d[thid + blockDim.x + blid * systemSize];

    __syncthreads();

    //forward elimination
    for (int j = 0; j <Iteration; j++)
    {
        __syncthreads();
	stride *= 2;
	int delta = stride/2;
        if (thid < thid_num)
	{ 
            int i = stride * thid + stride - 1;
            int iLeft = i - delta; 
            int iRight = i + delta;
            if (iRight >= systemSize) iRight = systemSize - 1;

            i = at32(i);
            iLeft = at32(iLeft);
            iRight = at32(iRight);

   	    T tmp1 = a[i] / b[iLeft];
	    T tmp2 = c[i] / b[iRight];
	    b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
	    d[i] = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
	    a[i] = -a[iLeft] * tmp1;
	    c[i] = -c[iRight]  * tmp2;
	}
        thid_num /= 2;
    }

    if (thid < 2)
    {
      int addr1 = stride - 1;
      int addr2 = 2 * stride - 1;

      addr1 = at32(addr1);
      addr2 = at32(addr2);
      
      T tmp3 = b[addr2]*b[addr1]-c[addr1]*a[addr2];
      x[addr1] = (b[addr2]*d[addr1]-c[addr1]*d[addr2])/tmp3;
      x[addr2] = (d[addr2]*b[addr1]-d[addr1]*a[addr2])/tmp3;
    }

    //backward substitution
    thid_num = 2;
    for (int j = 0; j <Iteration; j++)
    {
	int delta = stride/2;
	__syncthreads();
        if (thid < thid_num)
        {
            int i = stride * thid + stride/2 - 1;
            int iRight = i + delta;
            int iLeft = i - delta; 

            i = at32(i);
            iLeft = at32(iLeft);
            iRight = at32(iRight);

            if(i == delta - 1)
                x[i] = (d[i] - c[i]*x[iRight])/b[i];
            else
		x[i] = (d[i] - a[i]*x[iLeft] - c[i]*x[iRight])/b[i];
        }
	stride /= 2;
        thid_num *= 2;
    }

    __syncthreads();    

    x_d[thid + blid * systemSize] = x[at32(thid)];
    x_d[thid + blockDim.x + blid * systemSize] = x[at32(thid + blockDim.x)];

}

void crNBC32(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
{

    //printf("hello cyclic reduction\n");

    const unsigned int num_threads_block = systemSize/2;
    const unsigned int mem_size = sizeof(T)*numSystems*systemSize;

    // allocate device memory input and output arrays
    T* device_a;
    T* device_b;
    T* device_c;
    T* device_d;
    T* device_x;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_a,mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_b,mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_c,mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_d,mem_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &device_x,mem_size));

    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( device_a, a,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_b, b,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_c, c,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_d, d,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_x, x,mem_size, cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
	cutResetTimer(timer);
    cutStartTimer(timer);

	for(int i=0;i<numIterations;i++)
		crNBC32Kernel<<< grid, threads, systemSize*5*sizeof(T)*33/32>>>(device_a, device_b, device_c, device_d, device_x);

	cudaThreadSynchronize();
    cutStopTimer(timer);
	printf("crNBC: numSystems: %d, systemSize: %d, GPU kernel time: %f ms\n", numSystems, systemSize, cutGetTimerValue(timer)/numIterations);

	CUDA_SAFE_CALL( cudaMemcpy( device_a, a,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_b, b,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_c, c,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_d, d,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_x, x,mem_size, cudaMemcpyHostToDevice));
	crNBC32Kernel<<< grid, threads, systemSize*5*sizeof(T)*33/32>>>(device_a, device_b, device_c, device_d, device_x);

    // copy result from device to host
    //CUDA_SAFE_CALL( cudaMemcpy(a, device_a,mem_size, cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL( cudaMemcpy(b, device_b,mem_size, cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL( cudaMemcpy(c, device_c,mem_size, cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL( cudaMemcpy(d, device_d,mem_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL( cudaMemcpy(x, device_x,mem_size, cudaMemcpyDeviceToHost));

    // cleanup memory
    CUDA_SAFE_CALL(cudaFree(device_a));
    CUDA_SAFE_CALL(cudaFree(device_b));
    CUDA_SAFE_CALL(cudaFree(device_c));
    CUDA_SAFE_CALL(cudaFree(device_d));
    CUDA_SAFE_CALL(cudaFree(device_x));
}
#endif
