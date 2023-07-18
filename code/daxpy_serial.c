#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VECTOR_SIZE 100

int main( int argc, char* argv[] )
{

    int i;
    double alpha=1.0;
    // Number of bytes to allocate for N doubles
    size_t bytes = VECTOR_SIZE*sizeof(double);

    // Allocate memory for arrays X, A, B, and C on host
    double *X = (double*)malloc(bytes);
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    for(i=0;i<VECTOR_SIZE;i++){
        X[i]=M_PI*(double)(i+1)/VECTOR_SIZE;
        A[i]=B[i]=C[i]=0.0;
    }

    for(i=0;i<VECTOR_SIZE;i++){
        A[i]=cos(X[i])*cos(X[i]);
        B[i]=sin(X[i])*sin(X[i]);
        C[i]=alpha*A[i]-B[i];
    }

    if (VECTOR_SIZE<=100){
      for(i=0;i<VECTOR_SIZE;i++) printf("%9.5f %9.5f %9.5f %9.5f\n",X[i],A[i],B[i],C[i]);
    }
    else{
      for(i=VECTOR_SIZE-10;i<VECTOR_SIZE;i++) printf("%9.5f %9.5f %9.5f %9.5f\n",X[i],A[i],B[i],C[i]);
    }

    // Release memory
    free(A);
    free(B);
    free(C);

    return 0;
}
