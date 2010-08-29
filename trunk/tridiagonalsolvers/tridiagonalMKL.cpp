#include <omp.h>
#include <mkl_lapack.h>
#include "common.h"

void tridiagonalMKL(T *a, T *b, T *c, T * &d, T * &x, int system_size, int num_systems)
{
    T time_spent = 0;

    MKL_INT n = system_size;
    MKL_INT nrhs = 1;
    MKL_INT info;
    
    for (int i = 0; i < num_systems; i++)
	{
        #ifdef USE_DOUBLE
            dgtsv_(&n, &nrhs, &a[i*system_size+1], &b[i*system_size],&c[i*system_size],&d[i*system_size], &n, &info);
        #else
            sgtsv_(&n, &nrhs, &a[i*system_size+1], &b[i*system_size],&c[i*system_size],&d[i*system_size], &n, &info);
        #endif
	}

	T *tmp;
	tmp = x;
	x = d;
	d = tmp;
}

void multiThreadTridiagonalMKL(T *a, T *b, T *c, T * &d, T * &x, int system_size, int num_systems)
{
    T time_spent = 0;

    MKL_INT n = system_size;
    MKL_INT nrhs = 1;
    MKL_INT info;

    int numThreads = 4;
    int chunk = num_systems/numThreads;
    omp_set_num_threads(numThreads);

    #pragma omp parallel
    {
      int me = omp_get_thread_num();
      int start = me*chunk;
      int end   = start+chunk;
      for (int i = start; i < end; i++)
	  {
        #ifdef USE_DOUBLE
            dgtsv_(&n, &nrhs, &a[i*system_size+1], &b[i*system_size],&c[i*system_size],&d[i*system_size], &n, &info);
        #else
            sgtsv_(&n, &nrhs, &a[i*system_size+1], &b[i*system_size],&c[i*system_size],&d[i*system_size], &n, &info);
        #endif
	  }
	}

	T *tmp;
	tmp = x;
	x = d;
	d = tmp;
}
