#include "pcrGauss_kernel.cu"
#include "common.h"

void pcrGaussLarge(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems, int gaussian, int stride, int BPS)
{   
   // allocate device memory input and output arrays
   // setup execution parameters    
   dim3  grid(numSystems, 1, 1);
   dim3  threads(systemSize, 1, 1);
     
	
   switch(gaussian)
   {
		case 1:
			pcrKernelBranchFree<T, 1, 1, 0> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d,  numSystems, systemSize, stride, BPS);
			break;
		case 2:
			pcrKernelBranchFree<T, 1, 2, 1> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
			break;
		case 4:
			pcrKernelBranchFree<T, 1, 4, 2> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
			break;
		case 8:
			pcrKernelBranchFree<T, 1, 8, 2> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
			break;
		case 16:
			pcrKernelBranchFree<T, 1, 16, 2> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
			break;
		case 32:
			pcrKernelBranchFree<T, 1, 32, 2> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
			break;

       case 64:
           pcrKernelBranchFree<T, 1, 64, 2> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
           break;
           
       case 128:
           pcrKernelBranchFree <T, 1, 128, 3> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
		   break;
                
       case 256:
           pcrKernelBranchFree <T, 1, 256, 4> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
           break;
	   case 512:
           pcrKernelBranchFree <T, 1, 512, 5> <<< grid, threads,(systemSize+2)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize, stride, BPS);
           break;
   }
}
void pcrGaussSmall(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems, int gaussian, int stride, int BPS)
{   
    
	// allocate device memory input and output arrays      
    dim3  grid(numSystems, 1, 1);
    dim3  threads(systemSize, 1, 1);

    //pcrKernelBranchFreeOld<float> <<<grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, x, numSystems);
 
       switch(gaussian)
       {

		   case 1:
			   pcrKernelBranchFree <T, 1, 1, 0> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
	           break;
		   case 2:
			   pcrKernelBranchFree <T, 1, 2, 1> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
	           break;
		   case 4:
			   pcrKernelBranchFree <T, 1, 4, 2> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
	           break;
		   case 8:
			   pcrKernelBranchFree <T, 1, 8, 3> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
	           break;
		   case 16:
			   pcrKernelBranchFree <T, 1, 16, 3> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
	           break;
    
		   case 32:
			   pcrKernelBranchFree <T, 1, 32, 3> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
	           break;

		   case 64:
			   pcrKernelBranchFree<T, 1, 64, 3> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
			   break;
           
           case 128:
               pcrKernelBranchFree <T, 1, 128, 4> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
               break;
                
           case 256:
               pcrKernelBranchFree <T, 1, 256, 5> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
               break;

           case 512:
               pcrKernelBranchFree <T, 1, 512, 6> <<< grid, threads,(systemSize+1)*5*sizeof(T)>>>(a, b, c, d, numSystems, systemSize);
               break;
	   }
  
//          cutilCheckMsg("Kernel execution failed");
            
}

int solveLargeSystem(T *a, T *b, T *c, T *d, T *x, T *a2, T *b2, T *c2, T* d2, int smallSystem, int systemSize, int numSystems, int gaussian)
{
	int count = 0;
	int stride = 1;		
	//float *tmp = (float*)malloc(sizeof(float)* 5);
	cudaDeviceProp prop;
	int device_count;
	cudaGetDeviceCount( &device_count);

	int numSystemsSplit=systemSize;
	//printf("%d devices on this machine\n", device_count);

	cudaGetDeviceProperties( &prop, 0 );

	int numMPs = prop.multiProcessorCount;
	//printf("numSystems %d numMPS %d \n", numSystems, numMPs);
	int numMPsP2 = 1;
	while (numMPsP2 < numMPs)
		numMPsP2*=2;
	while(numSystemsSplit < numMPsP2)
	{
		if(count == 0)
		    globalPCROneStep<T, MINSYSTEM> <<<numMPsP2, smallSystem>>>((T *)a2, (T *)b2, (T *)c2, (T *)d2, (T *)a, (T *)b, 
											(T *)c,  (T *)d, systemSize, stride, numMPsP2/numSystems);             
		else
		    globalPCROneStep<T, MINSYSTEM> <<<numMPsP2, smallSystem>>>((T *)a, (T *)b, (T *)c, (T *)d, (T *)a2, (T *)b2, 
											(T *)c2,  (T *)d2, systemSize, stride, numMPsP2/numSystems);             
		numSystemsSplit*=2;		
		stride*=2;
		count++;
	}


	if(systemSize/stride > MINSYSTEM)
	{
	if(count == 0)
	    globalPCROneBlockSystem<T, MINSYSTEM> <<<numSystems, smallSystem>>> ((T *)a2, (T *)b2, (T *)c2, (T *)d2, (T *)a, (T *)b, 
											(T *)c,  (T *)d, systemSize, stride);             
	else
	    globalPCROneBlockSystem<T, MINSYSTEM> <<<numSystems, smallSystem>>> ((T *)a, (T *)b, (T *)c, (T *)d, (T *)a2, (T *)b2, 
											(T *)c2,  (T *)d2, systemSize, stride);             

		while(systemSize > smallSystem)
		{
            numSystems*=2;
            systemSize/=2;
            stride *=2;
			count+=1;
		}
					
     }		
		
	  if(count%2 == 0) //globalPCROneBlock... ping-pongs data between two buffers, count figures out which buffer was updated last
          pcrGaussLarge ((T *)a, (T *)b, (T *)c, (T *)d, 
							(T *)x, systemSize, numSystems, gaussian, stride, stride);
	  else
		  pcrGaussLarge ((T *)a2, (T *)b2, (T *)c2, (T *)d2, 
							(T *)x, systemSize, numSystems, gaussian, stride, stride);
	  //	CUDA_SAFE_CALL( cudaMemcpy( tmp, a,sizeof(float)*5, cudaMemcpyDeviceToHost));
	  return count;
}
	
void pcrGaussWrapper(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems, int gaussian)
{
	T* d_a;
    T* d_b;
    T* d_c;
    T* d_d;
    T* d_x;

	int count;
    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    cutStartTimer(timer);
	const unsigned int memSize = sizeof(T)*numSystems*systemSize;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c,memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d,memSize));
    //CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x,memSize));
	cutStopTimer(timer);
    
    printf("GPU cudaMalloc time: %f ms\n", cutGetTimerValue(timer));

   // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy( d_a, a,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_b, b,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_c, c,memSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL( cudaMemcpy( d_d, d,memSize, cudaMemcpyHostToDevice));
    //CUDA_SAFE_CALL( cudaMemcpy( d_x, x,memSize, cudaMemcpyHostToDevice));
    
	
	//can automate this part by using a device query to find out your
	//shared memory limit and deciding where to split from there
	if(systemSize > MINSYSTEM) // Fermi supports up to 1024 threads per block, i've set this to be defined in common.h
	{
		T* d_a2, *d_b2, *d_c2, *d_d2;
		CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a2,memSize));
	    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b2,memSize));
		CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c2,memSize));
		CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d2,memSize));		
		

		cutResetTimer(timer);
		cutStartTimer(timer);	
		for(int i =0;i<numIterations;i++)
		    solveLargeSystem(d_a, d_b,d_c, d_d, d_x, d_a2, d_b2, d_c2, d_d2, MINSYSTEM, systemSize, 
						numSystems, gaussian);
		cudaThreadSynchronize();
		cutStopTimer(timer);
		
		printf("pcrGauss: GPU kernel time: %f ms\n", cutGetTimerValue(timer)/numIterations);
		CUDA_SAFE_CALL( cudaMemcpy( d_a, a,memSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL( cudaMemcpy( d_b, b,memSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL( cudaMemcpy( d_c, c,memSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL( cudaMemcpy( d_d, d,memSize, cudaMemcpyHostToDevice));
	//	CUDA_SAFE_CALL( cudaMemcpy( d_x, x,memSize, cudaMemcpyHostToDevice));
		count = solveLargeSystem(d_a, d_b,d_c, d_d, d_x, d_a2, d_b2, d_c2, d_d2, MINSYSTEM, systemSize, 
		  		numSystems, gaussian);
		if(count%2 == 0)
		{
			CUDA_SAFE_CALL( cudaMemcpy(x, d_d,memSize, cudaMemcpyDeviceToHost));
		}
		else
		{
			CUDA_SAFE_CALL( cudaMemcpy(x, d_d2,memSize, cudaMemcpyDeviceToHost));
		}
		CUDA_SAFE_CALL(cudaFree(d_a2));
		CUDA_SAFE_CALL(cudaFree(d_b2));
		CUDA_SAFE_CALL(cudaFree(d_c2));
		CUDA_SAFE_CALL(cudaFree(d_d2));

		
	}
	else
	{
		cutResetTimer(timer);
		cutStartTimer(timer);
		for(int i =0;i<numIterations;i++)
		    pcrGaussSmall(d_a, d_b, d_c, d_d, d_x, systemSize, numSystems, gaussian, 1, 1);
		cudaThreadSynchronize();
		cutStopTimer(timer);
     	printf("pcrGauss: GPU kernel time: %f ms\n", cutGetTimerValue(timer)/numIterations);
		
		CUDA_SAFE_CALL( cudaMemcpy( d_a, a,memSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL( cudaMemcpy( d_b, b,memSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL( cudaMemcpy( d_c, c,memSize, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL( cudaMemcpy( d_d, d,memSize, cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL( cudaMemcpy( d_x, x,memSize, cudaMemcpyHostToDevice));
		pcrGaussSmall(d_a, d_b, d_c, d_d, d_x, systemSize, numSystems, gaussian, 1, 1);
		CUDA_SAFE_CALL( cudaMemcpy(x, d_d,memSize, cudaMemcpyDeviceToHost));
	}


    // copy result from device to host
   

	CUDA_SAFE_CALL(cudaFree(d_a));
	CUDA_SAFE_CALL(cudaFree(d_b));
	CUDA_SAFE_CALL(cudaFree(d_c));
	CUDA_SAFE_CALL(cudaFree(d_d));
	//CUDA_SAFE_CALL(cudaFree(d_x));


/** @} */ // end Parallel cyclic reduction solver (PCR)
/** @} */ // end cudpp_app

}