#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define TIMING_INFO

extern "C" 
{
#include "../../INCLUDE/f2c.h"
#include "../../INCLUDE/blaswrap.h"
#include "../../INCLUDE/clapack.h"
}

#include "file_read_write.c"
#include "test_gen_result_check.c"

void serial0(float *a,float *b,float *c,float *d,float *x,int num_elements)
{
    c[num_elements-1]=0;
    c[0]=c[0]/b[0];
    d[0]=d[0]/b[0];

    for (int i = 1; i < num_elements; i++)
    {
      c[i]=c[i]/(b[i]-a[i]*c[i-1]);
      d[i]=(d[i]-d[i-1]*a[i])/(b[i]-a[i]*c[i-1]);  
    }


    x[num_elements-1]=d[num_elements-1];

    for (int i = num_elements-2; i >=0; i--)
    {
      x[i]=d[i]-c[i]*x[i+1];
    }    
}

void serial(float *a,float *b,float *c,float *d,float *x,int num_elements)
{

    c[0]=c[0]/b[0];
    d[0]=d[0]/b[0];

    for (int i = 1; i < num_elements; i++)
    {
	   float tmp = 1/(b[i]-a[i]*c[i-1]);
      c[i]=c[i]*tmp;
      d[i]=(d[i]-d[i-1]*a[i])*tmp;  
    }


    x[num_elements-1]=d[num_elements-1];

    for (int i = num_elements-2; i >=0; i--)
    {
      x[i]=d[i]-c[i]*x[i+1];
    }
    
}

float lapack_small_systems(float *a, float *b, float *c, float *d, float *x, int system_size, int num_systems)
{

    float time_spent = 0;

    integer n=system_size;
    integer nrhs=1;
    integer info;

    
#ifdef TIMING_INFO 
    struct timeval T1,T2;
	gettimeofday(&T1,0);
    //clock_t t1, t2;
    //t1 = clock();
#endif

    for (int i = 0; i < num_systems; i++)
	{
        sgtsv_(&n, &nrhs, &a[i*system_size+1], &b[i*system_size],&c[i*system_size],&d[i*system_size], &n, &info);
 
        //serial0(&a[i*system_size],&b[i*system_size],&c[i*system_size],&d[i*system_size],&x[i*system_size],system_size);
	}

#ifdef TIMING_INFO 
    gettimeofday(&T2,0);
    time_spent = (float)(T2.tv_sec-T1.tv_sec)*1000+(float)(T2.tv_usec-T1.tv_usec)/1000;
    //t2 = clock();
    //time_spent = float(t2-t1)/float(CLOCKS_PER_SEC);

    printf("time spent: %f milliseconds\n",time_spent);
    //printf("time spent sec: %f seconds\n",(float)(T2.tv_sec-T1.tv_sec));
    //printf("time spent usec: %f microseconds\n",(float)(T2.tv_usec-T1.tv_usec));

 
#endif

    return time_spent;
}

float serial_small_systems(float *a, float *b, float *c, float *d, float *x, int system_size, int num_systems)
{

    float time_spent = 0;
 
#ifdef TIMING_INFO 
    struct timeval T1,T2;
	gettimeofday(&T1,0);
    //clock_t t1, t2;
    //t1 = clock();
#endif

    for (int i = 0; i < num_systems; i++)
	{
        serial(&a[i*system_size],&b[i*system_size],&c[i*system_size],&d[i*system_size],&x[i*system_size],system_size);
	}

#ifdef TIMING_INFO 
    gettimeofday(&T2,0);
    time_spent = (float)(T2.tv_sec-T1.tv_sec)*1000+(float)(T2.tv_usec-T1.tv_usec)/1000;
    //t2 = clock();
    //time_spent = float(t2-t1)/float(CLOCKS_PER_SEC);

    printf("time spent: %f milliseconds\n",time_spent);
    //printf("time spent sec: %f seconds\n",(float)(T2.tv_sec-T1.tv_sec));
    //printf("time spent usec: %f microseconds\n",(float)(T2.tv_usec-T1.tv_usec));

 
#endif

    return time_spent;
}

void runTest(int system_size,int num_systems,float *time_spent_gpu,float *time_spent_cpu,float *res)
{  
    const unsigned int mem_size = sizeof(float)*num_systems*system_size;

    float* a = (float*) malloc(mem_size);
    float* b = (float*) malloc(mem_size);
    float* c = (float*) malloc(mem_size);
    float* d = (float*) malloc(mem_size);
    float* x1 = (float*) malloc(mem_size);
    float* x2 = (float*) malloc(mem_size);

    for (int i = 0; i < num_systems; i++)
    {
        test_gen_cyclic(&a[i*system_size],&b[i*system_size],&c[i*system_size],&d[i*system_size],&x1[i*system_size],system_size,3);
        //test_gen_doubling(&a[i*system_size],&b[i*system_size],&c[i*system_size],&d[i*system_size],&x1[i*system_size],system_size,0);
	}

    float* a2 = (float*) malloc(mem_size);
    float* b2 = (float*) malloc(mem_size);
    float* c2 = (float*) malloc(mem_size);
    float* d2 = (float*) malloc(mem_size);
    array_backup(a2,a,system_size,num_systems);
    array_backup(b2,b,system_size,num_systems);
    array_backup(c2,c,system_size,num_systems);
    array_backup(d2,d,system_size,num_systems);

    *time_spent_cpu = serial_small_systems(a,b,c,d,x2,system_size,num_systems);    
    *res = residual_small_systems(a2,b2,c2,d2,x2,system_size,num_systems);

    //*time_spent_cpu = lapack_small_systems(a,b,c,d,x2,system_size,num_systems);
    //*res = residual_small_systems(a2,b2,c2,d2,d,system_size,num_systems);

    file_write_small_systems(d,10,system_size,"cpu.txt");

    //compare_small_systems(x1,x2,system_size,num_systems);

    free(a);
    free(b);
    free(c);
    free(d);
    free(x1);
    free(x2);
}

int main (int argc, char *argv[])
{
  int num_systems = 1;
  int system_size = 512;

  int ti=0;

  float time_gpu[9];
  float time_cpu[9];
  float res[9];

  //for(system_size=2;system_size<=512;system_size*=2)
  {
      num_systems = system_size;
      //num_systems = 1;
      runTest(system_size, num_systems,&time_gpu[ti],&time_cpu[ti],&res[ti]);
      printf("ti%d |num_systems=%d|system_size=%d|time_gpu=%f|time_cpu=%f\n",ti,num_systems,system_size,time_gpu[ti],time_cpu[ti]);
      ti++;
  }

  write_timing_results_1d(time_gpu,9,"timeGPU.txt");
  write_timing_results_1d(time_cpu,9,"timeCPU.txt");

  return EXIT_SUCCESS;
}
