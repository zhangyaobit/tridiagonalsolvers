#ifndef _CYCLIC_SMALL_SYSTEMS_NBC_
#define _CYCLIC_SMALL_SYSTEMS_NBC_

#include <stdio.h>
#include "common.h"

// at: address translater
__device__ int at(int x)
{
    int n = x >> 4;
    int m = x & 15;
    //printf("n = %d\n", n);
    //printf("m = %d\n", m);
    int y = n * 17 + m;
    //printf("y = %d\n", y);
    return y;
}

// cyclic reduction
__global__ void crNBCKernel(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d)
{
    int thid = threadIdx.x;	
    int blid = blockIdx.x;
    int stride = 1;
    int thid_num = blockDim.x;
    const unsigned int systemSize = blockDim.x * 2;
    int Iteration = (int)log2(float(systemSize/2));
    
    int systemSizeExt = systemSize * 17 / 16;
    extern __shared__ char shared[];
     T* a = (T*)shared;
    T* b = (T*)&a[systemSizeExt];
    T* c = (T*)&b[systemSizeExt];
    T* d = (T*)&c[systemSizeExt];
    T* x = (T*)&d[systemSizeExt];

    a[at(thid)] = a_d[thid + blid * systemSize];
    a[at(thid + blockDim.x)] = a_d[thid + blockDim.x + blid * systemSize];

    b[at(thid)] = b_d[thid + blid * systemSize];
    b[at(thid + blockDim.x)] = b_d[thid + blockDim.x + blid * systemSize];

    c[at(thid)] = c_d[thid + blid * systemSize];
    c[at(thid + blockDim.x)] = c_d[thid + blockDim.x + blid * systemSize];

    d[at(thid)] = d_d[thid + blid * systemSize];
    d[at(thid + blockDim.x)] = d_d[thid + blockDim.x + blid * systemSize];

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

            i = at(i);
            iLeft = at(iLeft);
            iRight = at(iRight);

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

      addr1 = at(addr1);
      addr2 = at(addr2);
      
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

            i = at(i);
            iLeft = at(iLeft);
            iRight = at(iRight);

            if(i == delta - 1)
                x[i] = (d[i] - c[i]*x[iRight])/b[i];
            else
		x[i] = (d[i] - a[i]*x[iLeft] - c[i]*x[iRight])/b[i];
        }
	stride /= 2;
        thid_num *= 2;
    }

    __syncthreads();    

    x_d[thid + blid * systemSize] = x[at(thid)];
    x_d[thid + blockDim.x + blid * systemSize] = x[at(thid + blockDim.x)];

}

void crNBC(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
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
		crNBCKernel<<< grid, threads, systemSize*5*sizeof(T)*17/16>>>(device_a, device_b, device_c, device_d, device_x);

	cudaThreadSynchronize();
    cutStopTimer(timer);
	printf("crNBC: numSystems: %d, systemSize: %d, GPU kernel time: %f ms\n", numSystems, systemSize, cutGetTimerValue(timer)/numIterations);

	CUDA_SAFE_CALL( cudaMemcpy( device_a, a,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_b, b,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_c, c,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_d, d,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_x, x,mem_size, cudaMemcpyHostToDevice));
	crNBCKernel<<< grid, threads, systemSize*5*sizeof(T)*17/16>>>(device_a, device_b, device_c, device_d, device_x);

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

__global__ void crpcrNBCKernel(T *a_d, T *b_d, T *c_d, T *d_d, T *x_d, int rest_system_size)
{
    int thid = threadIdx.x;	
    int blid = blockIdx.x;
    int stride = 1;
    int thid_num = blockDim.x;

    const unsigned int system_size = blockDim.x * 2;
    int Iteration = (int)log2(float(system_size/2));
    int restIteration = (int)log2(float(rest_system_size/2));

    int system_size_ext = system_size * 17 / 16;
    extern __shared__ char shared[];
    T* a = (T*)shared;
    T* b = (T*)&a[system_size_ext];
    T* c = (T*)&b[system_size_ext];
    T* d = (T*)&c[system_size_ext];
    T* x = (T*)&d[system_size_ext];

    a[at(thid)] = a_d[thid + blid * system_size];
    a[at(thid + blockDim.x)] = a_d[thid + blockDim.x + blid * system_size];

    b[at(thid)] = b_d[thid + blid * system_size];
    b[at(thid + blockDim.x)] = b_d[thid + blockDim.x + blid * system_size];

    c[at(thid)] = c_d[thid + blid * system_size];
    c[at(thid + blockDim.x)] = c_d[thid + blockDim.x + blid * system_size];

    d[at(thid)] = d_d[thid + blid * system_size];
    d[at(thid + blockDim.x)] = d_d[thid + blockDim.x + blid * system_size];

    __syncthreads();

    for (int j = 0; j <(Iteration-restIteration); j++)
    {
        stride *= 2;
        int delta = stride/2;

        if (thid < thid_num)
        { 
            int i = stride * thid + stride - 1;
            int iLeft = i - delta; 
            int iRight = i + delta;
            if (iRight >= system_size) iRight = system_size - 1;

            i = at(i);
            iLeft = at(iLeft);
            iRight = at(iRight);

   	        T tmp1 = a[i] / b[iLeft];
            T tmp2 = c[i] / b[iRight];
            b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
            d[i] = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
            a[i] = -a[iLeft] * tmp1;
            c[i] = -c[iRight]  * tmp2;
        }

        thid_num /= 2;
        __syncthreads();    
    }

    T* aa = (T*)&x[system_size_ext];
    T* bb = (T*)&aa[rest_system_size];
    T* cc = (T*)&bb[rest_system_size];
    T* dd = (T*)&cc[rest_system_size];
    T* xx = (T*)&dd[rest_system_size];

    if(thid < rest_system_size)
    {
        aa[thid] = a[at(thid*stride+stride-1)];
        bb[thid] = b[at(thid*stride+stride-1)];
        cc[thid] = c[at(thid*stride+stride-1)];
        dd[thid] = d[at(thid*stride+stride-1)];
    }
    
    int delta = 1;
    __syncthreads();

    //parallel cyclic reduction
    for (int j = 0; j <restIteration; j++)
    {
        int i = thid;
        T aNew, bNew, cNew, dNew;
        if(thid<rest_system_size)
        {         
            int iRight = i+delta;
            if (iRight >=  rest_system_size) iRight = rest_system_size-1;

            int iLeft = i-delta;
            if (iLeft < 0) iLeft = 0;
  
            T tmp1 = aa[i] / bb[iLeft];
            T tmp2 = cc[i] / bb[iRight];
            bNew = bb[i] - cc[iLeft] * tmp1 - aa[iRight] * tmp2;
            dNew = dd[i] - dd[iLeft] * tmp1 - dd[iRight] * tmp2;
            aNew = -aa[iLeft] * tmp1;
            cNew = -cc[iRight] * tmp2;
        }
        __syncthreads();
        
        if(thid<rest_system_size)	
        {
            bb[i] = bNew;
            dd[i] = dNew;
            aa[i] = aNew;
            cc[i] = cNew;	
        }
        delta *=2;
        __syncthreads();    
    }

    if (thid < delta)
    {
        int addr1 = thid;
        int addr2 = thid+delta;
        T tmp3 = bb[addr2]*bb[addr1]-cc[addr1]*aa[addr2];
        xx[addr1] = (bb[addr2]*dd[addr1]-cc[addr1]*dd[addr2])/tmp3;
        xx[addr2] = (dd[addr2]*bb[addr1]-dd[addr1]*aa[addr2])/tmp3;
    }
    __syncthreads(); 

    if(thid<rest_system_size)
        x[at(thid*stride+stride-1)]=xx[thid];

    //backward substitution
    thid_num = rest_system_size;
    for (int j = 0; j <(Iteration-restIteration); j++)
    {
        int delta = stride/2;
        __syncthreads();
        if (thid < thid_num)
        {
            int i = stride * thid + stride/2 - 1;
            int iRight = i + delta;
            int iLeft = i - delta; 

            i = at(i);
            iLeft = at(iLeft);
            iRight = at(iRight);

            if(i == delta - 1)
                x[i] = (d[i] - c[i]*x[iRight])/b[i];
            else
                x[i] = (d[i] - a[i]*x[iLeft] - c[i]*x[iRight])/b[i];
        }
        stride /= 2;
        thid_num *= 2;
    }

    __syncthreads();    

    x_d[thid + blid * system_size] = x[at(thid)];
    x_d[thid + blockDim.x + blid * system_size] = x[at(thid + blockDim.x)];
}

void crpcrNBC(T *a, T *b, T *c, T *d, T *x, int system_size, int num_systems, int rest_system_size)
{

    //printf("hello cyclic reduction\n");

    const unsigned int num_threads_block = system_size/2;
    const unsigned int mem_size = sizeof(T)*num_systems*system_size;

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
    dim3  grid(num_systems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);

    unsigned int timer;
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	cutResetTimer(timer);
    cutStartTimer(timer);

	for(int i = 0;i<numIterations;i++)
	    crpcrNBCKernel<<< grid, threads, system_size*5*sizeof(T)*17/16 + rest_system_size*(5+0)*sizeof(T)>>>(device_a, device_b, device_c, device_d, device_x, rest_system_size);

	cudaThreadSynchronize();
    cutStopTimer(timer);
	printf("crpcrNBC: numSystems: %d, systemSize: %d, GPU kernel time: %f ms\n", num_systems, system_size, cutGetTimerValue(timer)/numIterations);

    CUDA_SAFE_CALL( cudaMemcpy( device_a, a,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_b, b,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_c, c,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_d, d,mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( device_x, x,mem_size, cudaMemcpyHostToDevice));
    crpcrNBCKernel<<< grid, threads, system_size*5*sizeof(T)*17/16 + rest_system_size*(5+0)*sizeof(T)>>>(device_a, device_b, device_c, device_d, device_x, rest_system_size);

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
