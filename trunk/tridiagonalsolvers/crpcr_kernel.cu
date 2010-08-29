__global__ void crpcrKernel(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x, int restSystemSize)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int stride = 1;
    int numThreads = blockDim.x;
    const unsigned int systemSize = blockDim.x * 2;
    int iteration = (int)ceil(log2(T(systemSize/2)));
    //int restSystemSize = systemSize/2;
    int restIteration = (int)ceil(log2(T(restSystemSize/2)));

	#ifdef GPU_PRINTF
        if (thid == 0 && blid == 0) printf("iteration = %d; restIteration = %d\n", iteration, restIteration);
    #endif 
	//printf("dd");
    __syncthreads();

    extern __shared__ char shared[];
    T* a = (T*)shared;
    T* b = (T*)&a[systemSize+1];
    T* c = (T*)&b[systemSize+1];
    T* d = (T*)&c[systemSize+1];
    T* x = (T*)&d[systemSize+1];

    a[thid] = d_a[thid + blid * systemSize];
    a[thid + blockDim.x] = d_a[thid + blockDim.x + blid * systemSize];

    b[thid] = d_b[thid + blid * systemSize];
    b[thid + blockDim.x] = d_b[thid + blockDim.x + blid * systemSize];

    c[thid] = d_c[thid + blid * systemSize];
    c[thid + blockDim.x] = d_c[thid + blockDim.x + blid * systemSize];

    d[thid] = d_d[thid + blid * systemSize];
    d[thid + blockDim.x] = d_d[thid + blockDim.x + blid * systemSize];

    __syncthreads();

	// forward elimination in CR
    for (int j = 0; j <(iteration-restIteration); j++)
    {
        stride *= 2;
        int delta = stride/2;
        if (thid < numThreads)
		{
			int i = stride * threadIdx.x + stride - 1;
			int iLeft = i - delta;
			int iRight = i + delta;
			if (iRight >= systemSize) iRight = systemSize - 1;
			T tmp1 = a[i] / b[iLeft];
			T tmp2 = c[i] / b[iRight];
			b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
			d[i] = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
			a[i] = -a[iLeft] * tmp1;
			c[i] = -c[iRight] * tmp2;
		}
        numThreads /= 2;
        __syncthreads();    
    }
    
	// solve the intermediate system with PCR
    T* aa = (T*)&x[systemSize+1];
    T* bb = (T*)&aa[restSystemSize];
    T* cc = (T*)&bb[restSystemSize];
    T* dd = (T*)&cc[restSystemSize];
    T* xx = (T*)&dd[restSystemSize];

    if(thid<restSystemSize)
    {
        aa[thid] = a[thid*stride+stride-1];
        bb[thid] = b[thid*stride+stride-1];
        cc[thid] = c[thid*stride+stride-1];
        dd[thid] = d[thid*stride+stride-1];
	}

    T aNew, bNew, cNew, dNew;
    int delta = 1;
    __syncthreads();
//*
    //parallel cyclic reduction
    for (int j = 0; j <restIteration; j++)
    {
		int i = thid;
		if(thid < restSystemSize)
		{
			int iRight = i+delta;
			if (iRight >= restSystemSize) iRight = restSystemSize - 1;

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
        if(thid<restSystemSize)
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
	if(thid < restSystemSize)
        x[thid*stride+stride-1]=xx[thid];
    
    // backward substitution in CR
    numThreads = restSystemSize;
    
    for (int j = 0; j <(iteration-restIteration); j++)
    {
        int delta = stride/2;
        __syncthreads();
        if (thid < numThreads)
        {
            int i = stride * thid + stride/2 - 1;
            if(i == delta - 1)
            x[i] = (d[i] - c[i]*x[i+delta])/b[i];
            else
            x[i] = (d[i] - a[i]*x[i-delta] - c[i]*x[i+delta])/b[i];
        }
        stride /= 2;
        numThreads *= 2;
    }

    __syncthreads();    
//*/

    d_x[thid + blid * systemSize] = x[thid];
    d_x[thid + blockDim.x + blid * systemSize] = x[thid + blockDim.x];
}
