#include <stdio.h>

// Size of array
#define VECTOR_SIZE 100

// Kernel
__global__ void add_vectors(double alpha, double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < VECTOR_SIZE) c[id] = a[id] - b[id];
}

// Main program
int main( int argc, char* argv[] )
{
    double alpha=1.0;
    // Number of bytes to allocate for N doubles
    size_t bytes = VECTOR_SIZE*sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double *X = (double*)malloc(bytes);
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Fill host arrays A and B
    for(int i=0; i<VECTOR_SIZE; i++)
    {
        X[i]=M_PI*(double)(i+1)/VECTOR_SIZE;
        A[i]=cos(X[i])*cos(X[i]);
        B[i]=sin(X[i])*sin(X[i]);
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 32;
    int blk_in_grid = ceil( double(VECTOR_SIZE) / thr_per_blk );

    // Launch kernel
    add_vectors<<< blk_in_grid, thr_per_blk >>>(alpha, d_A, d_B, d_C);

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    for(int i=0; i<VECTOR_SIZE; i++)
    {
        if(C[i] != alpha*A[i]-B[i])
        {
            printf("\nError: value of C[%d] = %f instead of %f\n\n", i, C[i], alpha*A[i]-B[i]);
            //exit(-1);
        }
        else
        {
            printf("X=%f C=%f\n",X[i],C[i]);
        }
    }

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("VECTOR SIZE       = %d\n", VECTOR_SIZE);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
}

