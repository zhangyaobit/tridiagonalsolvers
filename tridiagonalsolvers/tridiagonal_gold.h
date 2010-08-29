// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * tridiagonal_gold.cpp
 *
 * @brief Host testrig routines for the tridiagonal solver
 */

#include <omp.h>
#include <stdio.h>
#include "common.h"
#include <iostream>

using namespace std;

//template <class T>
void serial(T *a, T *b, T *c, T *d, T *x, int numElements)
{
    c[numElements - 1] = 0;
    c[0] = c[0] / b[0];
    d[0] = d[0] / b[0];

    for (int i = 1; i < numElements; i++)
    {
      c[i] = c[i] / (b[i] - a[i] * c[i-1]);
      d[i] = (d[i] - d[i-1] * a[i]) / (b[i] - a[i] * c[i-1]);
    }

    x[numElements-1] = d[numElements-1];

    for (int i = numElements-2; i >= 0; i--)
    {
      x[i] = d[i] - c[i] * x[i+1];
    }
}

//template <class T>
void serialManySystems(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
{
    for (int i = 0; i < numSystems; i++)
    {
        serial(&a[i*systemSize], &b[i*systemSize], &c[i*systemSize], &d[i*systemSize], &x[i*systemSize],systemSize);
    }
}

//template <class T>
void multiThreadManySystems(T *a, T *b, T *c, T *d, T *x, int systemSize, int numSystems)
{
    int nunThreads = 4;
    int chunk = numSystems/nunThreads;
    omp_set_num_threads(nunThreads);
    
    #pragma omp parallel
    {
      int me = omp_get_thread_num();
      int start = me*chunk;
      int end   = start+chunk;
      for (int i = start; i < end; i++)
	  {
        serial(&a[i*systemSize],&b[i*systemSize],&c[i*systemSize],&d[i*systemSize],&x[i*systemSize],systemSize);
	  }
	}
}

//template <class T>
T rand01()
{
    return T(rand()) / T(RAND_MAX);
}

//template <class T>
void testGeneration(T *a, T *b, T *c, T *d, T *x, int systemSize)
{
    //generate a diagonally dominated matrix
    for (int j = 0; j < systemSize; j++)
    {
        b[j] = 8 + rand01();
        a[j] = 3 + rand01();
        c[j] = 2 + rand01();
        d[j] = 5 + rand01();
        x[j] = 0;
    }
    a[0] = 0;
    c[systemSize-1] = 0;	
}

//template <class T>
void writeResultToFile(T *x,int numSystems,int systemSize, char *file_name)
{
    ofstream myfile;
    myfile.open (file_name);
    for(int i=0; i < numSystems * systemSize; i++)
    {
        if (i%systemSize==0)
            myfile << "***The following is the result of the equation set " << i/systemSize << "\n";
        myfile << x[i] << "\n";
    }
    myfile.close();
}

//template <class T>
T compare(T *x1, T *x2, int numElements, int systemNum)
{
    T mean = 0;//mean error
    T root = 0;//root mean square error
    T maxer = 0; //max error

    for (int i = 0; i < numElements; i++)
    {
        root += (x1[i] - x2[i]) * (x1[i] - x2[i]);
        mean += fabs(x1[i] - x2[i]);
        if(fabs(x1[i] - x2[i]) > maxer) maxer = fabs(x1[i] - x2[i]);

    }
    mean /= numElements;
    root /= numElements;
    root = sqrt(root); 

	
    return root;
}

//template <class T>
int compareManySystems(T *x1, T *x2, int systemSize, int numSystems, const T epsilon)
{
    int retval = 0;

    for (int i = 0; i < numSystems; i++)
    {
        T diff = compare(&x1[i*systemSize], &x2[i*systemSize], systemSize, i);
        if(diff > epsilon || diff != diff) //if diff is QNAN/NAN, diff != diff will return true
        {
            cout << "test failed for "<< i << " error is " << diff << "\n";
            retval++;
        }
    }

    return retval;
}
