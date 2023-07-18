#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MASTER 0
#define VECTOR_SIZE 100

int main (int argc, char *argv[]) 
{
    double alpha=1.0;
    double * A;
    double * B; 
    double * C;
    double * X;

    // arrays a and b
    int total_proc; 
    // total nuber of processes 
    int rank;
    // rank of each process
    int T;
    // total number of test cases
    long long int n_per_proc;   
    // elements per process     

    long long int i, j, n;
    MPI_Status status;

    // Initialization of MPI environment
    MPI_Init (&argc, &argv);

    MPI_Comm_size (MPI_COMM_WORLD, &total_proc); //The total number of processes running in parallel
    MPI_Comm_rank (MPI_COMM_WORLD, &rank); //The rank of the current process

    double * x;
    double * ap;
    double * bp;
    double * cp;
    double * buf;

    n_per_proc = VECTOR_SIZE/total_proc;
    if(VECTOR_SIZE%total_proc != 0) n_per_proc+=1; // to divide data evenly by the number of processors 

    x  = (double *) malloc(sizeof(double)*n_per_proc);
    ap = (double *) malloc(sizeof(double)*n_per_proc);
    bp = (double *) malloc(sizeof(double)*n_per_proc);
    cp = (double *) malloc(sizeof(double)*n_per_proc);
    buf = (double *) malloc(sizeof(double)*n_per_proc);

    for(i=0;i<n_per_proc;i++){
        //printf("i=%d\n",i);
        x[i]=M_PI*(rank*n_per_proc+i)/VECTOR_SIZE;
        ap[i]=cos(x[i])*cos(x[i]);
        bp[i]=sin(x[i])*sin(x[i]);
        cp[i]=alpha*ap[i]-bp[i];
    }

    if (rank == MASTER){

        for(i=0;i<total_proc;i++){
            
            if(i==MASTER){
                printf("Skip\n");
            }
            else{
                printf("Receiving from %d...\n",i);
                MPI_Recv(buf, n_per_proc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                for(j=0;j<n_per_proc;j++) printf("rank=%d i=%d x=%f c=%f\n",i,j,M_PI*(i*n_per_proc+j)/VECTOR_SIZE,buf[j]);
            }
        }
    }
    else{
        MPI_Bsend(cp, n_per_proc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    //Terminate MPI Environment
    return 0;
}

