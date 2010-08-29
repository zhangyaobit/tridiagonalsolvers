__global__ void pcrKernel(T *d_a, T *d_b, T *d_c, T *d_d, T *d_x)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int delta = 1;
    const unsigned int systemSize = blockDim.x;
    int iteration = (int)log2(T(systemSize/2));

    __syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[systemSize+1];
    T* c = (T*)&b[systemSize+1];
    T* d = (T*)&c[systemSize+1];
    T* x = (T*)&d[systemSize+1];

    a[thid] = d_a[thid + blid * systemSize];
    b[thid] = d_b[thid + blid * systemSize];
    c[thid] = d_c[thid + blid * systemSize];
    d[thid] = d_d[thid + blid * systemSize];
  
    //T aNew, bNew, cNew, dNew;
  
    __syncthreads();

    //parallel cyclic reduction
    for (int j = 0; j <iteration; j++)
    {
        int i = thid;

        int iRight = i+delta;
        if (iRight >= systemSize) iRight = systemSize - 1;

        int iLeft = i-delta;
        if (iLeft < 0) iLeft = 0;

        T tmp1 = a[i] / b[iLeft];
        T tmp2 = c[i] / b[iRight];
		T tmp3 = d[iRight];
		T tmp4 = d[iLeft];
		__syncthreads();

        //bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        //dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
        //aNew = -a[iLeft] * tmp1;
        //cNew = -c[iRight] * tmp2;

		b[i] = b[i]-c[iLeft]*tmp1-a[iRight]*tmp2;
        d[i] = d[i] - tmp4*tmp1-tmp3*tmp2;
		tmp3 = -a[iLeft];
		tmp4 = -c[iRight];
		__syncthreads();
		a[i] = tmp3*tmp1;
		c[i] = tmp4*tmp2;
        
		__syncthreads();
        
              
        delta *=2;   
    }

    if (thid < delta)
    {
		
        int addr1 = thid;
        int addr2 = thid+delta;
        T tmp3 = b[addr2]*b[addr1]-c[addr1]*a[addr2];
        x[addr1] = (b[addr2]*d[addr1]-c[addr1]*d[addr2])/tmp3;
        x[addr2] = (d[addr2]*b[addr1]-d[addr1]*a[addr2])/tmp3;
    }

    __syncthreads();

    d_x[thid + blid * systemSize] = x[thid];
}
