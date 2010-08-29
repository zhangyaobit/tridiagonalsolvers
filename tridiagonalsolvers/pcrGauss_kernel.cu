template <class T, int gaussElim>
__device__ void serial(T *a, T *b, T *c, T *d, int numEqs, int thid){   

    c[thid] = c[thid] / b[thid]; 
    d[thid] = d[thid] / b[thid];
    T tmp1, tmp2;

    for (int i = gaussElim+thid; i < numEqs; i+=gaussElim) //i=stride+thid; i + = stride; i < numEqs*numSerialize (systemSize)
    {
		//iterations = numSystems/gaussElim - 1
	  //5 shared memory loads, 2 shared memory store
	  tmp2 = a[i];
      tmp1 = (b[i] - tmp2 * c[i-gaussElim]);
      c[i] /= tmp1;//(b[i] - a[i] * c[i-stride]);
      d[i] = (d[i] - d[i-gaussElim] * tmp2) /tmp1;//(b[i] - a[i] * c[i-stride]); //i - stride
	  
    }

    //x[numEqs-gaussElim+thid] = d[numEqs-gaussElim+thid]; //c[(numEqs-1)*numThreads+thid]

	tmp1 = d[numEqs-gaussElim+thid];
    for (int i =numEqs-2*gaussElim+thid; i >= 0; i-=gaussElim) //c[(numEqs-2)*numThreads+thid] i-=stride
    {
		//iterations = numSystems/gaussElim - 1
		//3 loads, 1 store
      tmp1 = d[i] - c[i] * tmp1; //i + stride
	  d[i] = tmp1;
    }

	//total loads: 7*(numSystems/gaussElim-1) + 4
	//total stores 3*(numSystems/gaussElim-1) + 2
}
/*template <class T, int gaussElim>
__device__ void serial(T *a, T *b, T *c, T *d, int numEqs, int thid){   

    //c[(numEqs-1)*stride + thid] = 0;
    c[thid] = c[thid] / b[thid]; // c[thid]
    d[thid] = d[thid] / b[thid];
    T tmp1;

    for (int i = gaussElim+thid; i < numEqs; i+=gaussElim) //i=stride+thid; i + = stride; i < numEqs*numSerialize (systemSize)
    {
      tmp1 = (b[i] - a[i] * c[i-gaussElim]);
      c[i] = c[i] / tmp1;//(b[i] - a[i] * c[i-stride]);
      d[i] = (d[i] - d[i-gaussElim] * a[i]) /tmp1;//(b[i] - a[i] * c[i-stride]); //i - stride
    }

    //x[numEqs-gaussElim+thid] = d[numEqs-gaussElim+thid]; //c[(numEqs-1)*numThreads+thid]

    for (int i =numEqs-2*gaussElim+thid; i >= 0; i-=gaussElim) //c[(numEqs-2)*numThreads+thid] i-=stride
    {
      d[i] = d[i] - c[i] * d[i+gaussElim]; //i + stride
    }
}*/

template <class T, int loopsPerThread, int gaussElim, int iteration>
__global__ void pcrKernelBranchFree(T *d_a, T *d_b, T *d_c, T *d_d, int numEquations, int systemSize)
{
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    
    int delta = 1;        


    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[systemSize+1];
    T* c = (T*)&b[systemSize+1];
    T* d = (T*)&c[systemSize+1];
    
    T aNew, bNew, cNew, dNew;  
    
	 
	a[thid] = d_a[thid + blid * systemSize];	
	b[thid] = d_b[thid + blid * systemSize];	
	c[thid] = d_c[thid + blid * systemSize];	
	d[thid] = d_d[thid + blid * systemSize];
			
	__syncthreads();

    for (int j = 0; j <iteration; j++)
    {    
		//12 loads and four saves per iteration
		//iteration = log2(gaussElim)
		int iRight = min(thid+delta, systemSize-1);
		int iLeft = max(thid-delta, 0);

        T tmp1 = a[thid]/b[iLeft];
		aNew = -a[iLeft]*tmp1;

		bNew = b[thid] - c[iLeft]*tmp1;		
		dNew = d[thid] - d[iLeft]*tmp1;

	    tmp1 = c[thid]/b[iRight];				
		bNew -= a[iRight]*tmp1;
		cNew = -c[iRight]*tmp1;			
		dNew -= d[iRight]*tmp1;				     

      
	    __syncthreads();
		a[thid] = aNew;
        b[thid] = bNew;
		c[thid] = cNew;
        d[thid] = dNew;
        
        
		__syncthreads();
		delta = delta* 2;
   }    

    if(thid<gaussElim)
    {
        serial<T,gaussElim>(a,b,c,d,systemSize, thid);        
    }
    __syncthreads();

    //Total Shared Mem loads per thread: 5 + 12*log2(gaussElim) + gaussElim/numThreads*(8*numThreads/gaussElim-1 + 5)
	//Total Shared Mem loads per thread: 5 + 12*log2(gaussElim) + (8+4*gaussElim/numThreads) = 13 + 12*log2(gaussElim) + 4*gaussElim/numThreads

	//Total Shared Mem stores per thread: 4*log2(gaussElim) + (3+3*gaussElim/numThreads) = 3 + 4*log2(gaussElim) + 3*gaussElim/numThreads 

//#pragma unroll
    d_d[thid + blid * systemSize] = d[thid];

}

template <class T, int loopsPerThread, int gaussElim, int iteration>
__global__ void pcrKernelBranchFree(T *d_a, T *d_b, T *d_c, T *d_d, int numEquations, int systemSize, int extraStride, int BPS)
{   
    int thid = threadIdx.x;
    int blid = blockIdx.x;
    int bdim = blockDim.x;
    int delta = extraStride;
	int bside = blid%BPS;
    int l = 0;
    
    //int iteration = (int)log2(T(gaussElim));       

    //__syncthreads();

    extern __shared__ char shared[];

    T* a = (T*)shared;
    T* b = (T*)&a[systemSize+2];
    T* c = (T*)&b[systemSize+2];
    T* d = (T*)&c[systemSize+2];
    T* x = (T*)&d[systemSize+2];


    a[thid] = d_a[(thid + blid * systemSize)];
    b[thid] = d_b[(thid + blid * systemSize)];
    c[thid] = d_c[(thid+ blid * systemSize)];
    d[thid] = d_d[(thid+ blid * systemSize)];

    T aNew, bNew, cNew, dNew;
	
    __syncthreads();	    

    //parallel cyclic reduction
    for (int j = 0; j <iteration; j++)
    {
    //#pragma unroll
       int i = thid;  
       int iRight = i+delta;

       if (iRight >= systemSize && bside==BPS-1) 
			iRight = systemSize - 1;

       int iLeft = i-delta;
	   if (iLeft < 0 && bside == 0) iLeft = 0;
	   if(iRight >=systemSize)
		{
			bNew = b[i] - c[iLeft]*a[i]/b[iLeft] - d_a[blid*systemSize + iRight]*c[i]/d_b[blid*systemSize+ iRight];
			dNew = d[i] - d[iLeft]*a[i]/b[iLeft] - d_d[blid*systemSize + iRight]*c[i]/d_b[blid*systemSize + iRight];			
			aNew = -a[iLeft] * a[i]/b[iLeft];
			cNew = -d_c[blid*systemSize+iRight]*c[i]/d_b[blid*systemSize+iRight];
		}
	        
	       //iLeft = iLeft%systemSize;	        
		else if(iLeft < 0)
		{
		    bNew = b[i] - d_c[blid*systemSize+iLeft] * a[i]/d_b[blid*systemSize+iLeft] - a[iRight] * c[i]/b[iRight];
	        dNew = d[i] - d_d[blid*systemSize+iLeft] * a[i]/d_b[blid*systemSize+iLeft] - d[iRight] * c[i]/b[iRight];
	        aNew = -d_a[blid*systemSize+iLeft] * a[i]/d_b[blid*systemSize+iLeft];
	        cNew = -c[iRight] * c[i]/b[iRight];	      
		}

		else
		{
	        T tmp1 = a[i] / b[iLeft];
	        T tmp2 = c[i] / b[iRight];

	        bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
	        dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
	        aNew = -a[iLeft] * tmp1;
	        cNew = -c[iRight] * tmp2;	      
		}
	delta *= 2;	
	__syncthreads();
	
      

   	b[thid] = bNew;
	d[thid] = dNew;
	a[thid] = aNew;
	c[thid] = cNew;   
	
	
   }    

    if(thid<gaussElim)
    {
        serial<T,gaussElim>(a,b,c,d,systemSize, thid);        
    }
    __syncthreads();


//#pragma unroll
    for(l=0; l < loopsPerThread; l++)
	{ 
    d_d[thid + l*bdim + blid * systemSize] = d[thid+l*bdim];
    }
}

template <class T, int smallSystem>
__global__ void globalPCROneStep(T* d_aout, T* d_bout, T* d_cout, T*d_dout, 
								 T *d_a, T *d_b, T *d_c, T *d_d, int systemSize, 
								 int stride, int numBlocksPerSystem)
{

    int thid = threadIdx.x;
    int blid = blockIdx.x;    
	//int bSide = blid & (numBlocksPerSystem-1);

	//int delta = stride;
    int bSide = blid%numBlocksPerSystem;


    //__syncthreads();
    //parallel cyclic reduction
	int blockLeftMin = int(blid/numBlocksPerSystem)*systemSize;
	int blockRightMax = blockLeftMin + systemSize;
	int pos = bSide*smallSystem + thid;
	int i = thid + blockLeftMin + bSide*blockDim.x;
	while(pos<systemSize)
	{		
		
		int iRight = i+stride;    

		if (iRight >= blockRightMax) iRight = blockRightMax-1;

		int iLeft = i-stride;
    //iLeft = iLeft%systemSize;
		if (iLeft < blockLeftMin) iLeft = blockLeftMin;

		T tmp1 = d_a[i] / d_b[iLeft];
		T tmp2 = d_c[i] / d_b[iRight];

	//could get small reuse (3x?) if i still share between threads
		d_bout[i] = d_b[i] - d_c[iLeft] * tmp1 - d_a[iRight] * tmp2;
		d_dout[i] = d_d[i] - d_d[iLeft] * tmp1 - d_d[iRight] * tmp2;
		d_aout[i] = -d_a[iLeft] * tmp1;
		d_cout[i] = -d_c[iRight] * tmp2;                   
		blockLeftMin += blockDim.x*gridDim.x;
		pos += smallSystem*numBlocksPerSystem;
		i += smallSystem*numBlocksPerSystem;
	}
//need global synch here?
}

template <class T, int smallSystem>
__global__ void globalPCROneBlockSystem(T* d_aout, T* d_bout, T* d_cout, T*d_dout, 
								 T *d_a, T *d_b, T *d_c, T *d_d, int systemSize, 
								 int stride)
{

    int thid = threadIdx.x;
    int blid = blockIdx.x;    	


	int tid;
	int delta = 1;
#if USESM
/*	extern __shared__ char shared[];
	T* a = (T*)shared;
    T* b = (T*)&a[systemSize+2];
    T* c = (T*)&b[systemSize+2];
    T* d = (T*)&c[systemSize+2];*/
#endif

	int blockLeftMin = systemSize*blockIdx.x;
	int blockRightMax = blockLeftMin + systemSize;

	int count = 0;
	while(systemSize/delta > smallSystem)
	{
		tid = blockLeftMin+thid;
		while(tid < blockRightMax)
		{		
			int i = tid;
			int iRight = i+delta;

			if (iRight >= blockRightMax) iRight = blockRightMax-1;

			int iLeft = i-delta;
		//iLeft = iLeft%systemSize;
			if (iLeft < blockLeftMin) iLeft = blockLeftMin;

			if(count == 0)
			{
				
				//experimenting to see if I can get some speedup using shared memory
				//for a little bit of data-reuse
				//this might be a machine dependent parameter that might not be useful for
				//fermi cards, but useful for non-caching global memory
#if USESM
			/*	a[thid] = d_a[i];
				b[thid] = d_b[i];
				c[thid] = d_c[i];
				d[thid] = d_d[i];
				
				T tmp1 = a[thid]/d_b[iLeft]; 
				T tmp2 = c[thid]/d_b[iRight];

	
				d_bout[i] =  b[thid]-d_c[iLeft]*tmp1-d_a[iRight]*tmp2; //
				d_dout[i] =  d[thid]-d_d[iLeft]*tmp1-d_d[iRight]*tmp2; //
				d_aout[i] = -d_a[iLeft] * tmp1; //
				d_cout[i] = -d_c[iRight] * tmp2; // 					*/
#else		


				T tmp1 =  d_a[i] / d_b[iLeft]; //a[thid]/d_b[iLeft]; //
				T tmp2 =  d_c[i] / d_b[iRight]; //c[thid]/d_b[iRight]; //

	//could get small reuse (3x?) if i still share between threads
				d_bout[i] =  d_b[i] - d_c[iLeft] * tmp1 - d_a[iRight] * tmp2; //b[thid]-d_c[iLeft]*tmp1-d_a[iRight]*tmp2; //
				d_dout[i] =  d_d[i] - d_d[iLeft] * tmp1 - d_d[iRight] * tmp2; //d[thid]-d_d[iLeft]*tmp1-d_d[iRight]*tmp2; //
				d_aout[i] = -d_a[iLeft]*tmp1;// -d_a[iLeft] * tmp1; //
				d_cout[i] =  -d_c[iRight]*tmp2;//-d_c[iRight] * tmp2; // 					
#endif
            
			}
			else
			{
#if USESM
		/*		a[thid] = d_aout[i];
				b[thid] = d_bout[i];
				c[thid] = d_cout[i];
				d[thid] = d_dout[i];
				
				T tmp1 = a[thid]/d_bout[iLeft]; 
				T tmp2 = c[thid]/d_bout[iRight];

	
				d_bout[i] =  b[thid]-d_c[iLeft]*tmp1-d_aout[iRight]*tmp2; //
				d_dout[i] =  d[thid]-d_d[iLeft]*tmp1-d_dout[iRight]*tmp2; //
				d_aout[i] = -d_aout[iLeft] * tmp1; //
				d_cout[i] = -d_cout[iRight] * tmp2; // 					*/
#else		


				T tmp1 =  d_aout[i] / d_bout[iLeft]; //a[thid]/d_b[iLeft]; //
				T tmp2 =  d_cout[i] / d_bout[iRight]; //c[thid]/d_b[iRight]; //

	//could get small reuse (3x?) if i still share between threads
				d_b[i] =  d_bout[i] - d_cout[iLeft] * tmp1 - d_aout[iRight] * tmp2; //b[thid]-d_c[iLeft]*tmp1-d_a[iRight]*tmp2; //
				d_d[i] =  d_dout[i] - d_dout[iLeft] * tmp1 - d_dout[iRight] * tmp2; //d[thid]-d_d[iLeft]*tmp1-d_d[iRight]*tmp2; //
				d_a[i] = -d_aout[iLeft]*tmp1;// -d_a[iLeft] * tmp1; //
				d_c[i] =  -d_cout[iRight]*tmp2;//-d_c[iRight] * tmp2; // 					
#endif            				
			}
		tid += blockDim.x;
		__syncthreads();
			
		}
	  
		delta*=2;
		if(count == 0)
			count = 1;
		else
			count = 0;
	}

}
