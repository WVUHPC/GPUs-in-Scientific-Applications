#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define VECTOR_SIZE 100

void skip(int i) {}
void work(int i) {}

int main( int argc, char* argv[] ){

    int i,id;
    double alpha=1.0;

    // Number of bytes to allocate for N doubles
    size_t bytes = VECTOR_SIZE*sizeof(double);

    // Allocate memory for arrays X, A, B, and C on host
    double *X = (double*)malloc(bytes);
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    omp_lock_t lck;
    omp_init_lock(&lck);

    #pragma omp parallel for shared(X,A,B,lck) private(i,id)
    for(i=0;i<VECTOR_SIZE;i++){
        id = omp_get_thread_num();

        omp_set_lock(&lck);
        /*  only one thread at a time can execute this printf */
        printf("My thread id is %d working on i=%d\n", id,i);
        omp_unset_lock(&lck);

        while (! omp_test_lock(&lck)) {
            skip(id);   /* we do not yet have the lock,
                         so we must do something else */
        }

        work(id);      /* we now have the lock
                          and can do the work */

        omp_unset_lock(&lck); 
        X[i]=M_PI*(double)(i+1)/VECTOR_SIZE;
        A[i]=cos(X[i])*cos(X[i]);
        B[i]=sin(X[i])*sin(X[i]);
  }

  #pragma omp parallel for shared(alpha,A,B,C,lck) private(i,id) schedule(static,10)
  for(i=0;i<VECTOR_SIZE;i++){
    id = omp_get_thread_num();

    omp_set_lock(&lck);
    /*  only one thread at a time can execute this printf */
    printf("My thread id is %d working on i=%d\n", id,i);
    omp_unset_lock(&lck);

    while (! omp_test_lock(&lck)) {
      skip(id);   /* we do not yet have the lock,
                     so we must do something else */
    }

    work(id);      /* we now have the lock
                      and can do the work */

    omp_unset_lock(&lck);
 
    C[i]=alpha*A[i]-B[i];
    printf("i=%d X=%f alpha*A-B=%f\n",i,X[i],C[i]);
  }
  omp_destroy_lock(&lck);

  return 0;
}

