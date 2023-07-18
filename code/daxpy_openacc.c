#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VECTOR_SIZE 100

int main( int argc, char* argv[] )
{
 
    double alpha=1.0;
    double *restrict X;
    // Input vectors
    double *restrict A;
    double *restrict B;
    // Output vector
    double *restrict C;
 
    // Size, in bytes, of each vector
    size_t bytes = VECTOR_SIZE*sizeof(double);
 
    // Allocate memory for each vector
    X = (double*)malloc(bytes);
    A = (double*)malloc(bytes);
    B = (double*)malloc(bytes);
    C = (double*)malloc(bytes);
 
    // Initialize content of input vectors, vector A[i] = cos(i)^2 vector B[i] = sin(i)^2
    int i;
    for(i=0; i<VECTOR_SIZE; i++) {
        X[i]=M_PI*(double)(i+1)/VECTOR_SIZE;
        A[i]=cos(X[i])*cos(X[i]);
        B[i]=sin(X[i])*sin(X[i]);
    }   
 
    // sum component wise and save result into vector c
    #pragma acc kernels copyin(A[0:VECTOR_SIZE],B[0:VECTOR_SIZE]), copyout(C[0:VECTOR_SIZE])
    for(i=0; i<VECTOR_SIZE; i++) {
        C[i] = alpha*A[i]-B[i];
    }

    for(i=0; i<VECTOR_SIZE; i++) printf("X=%9.5f C=%9.5f\n",X[i],C[i]);
 
    // Release memory
    free(A);
    free(B);
    free(C);
 
    return 0;
}

