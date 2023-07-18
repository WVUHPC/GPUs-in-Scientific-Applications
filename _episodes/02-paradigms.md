---
title: "Paradigms of Parallel Computing"
teaching: 30
exercises: 0
questions:
- "Which are the ways we can use parallel computing?"
- "On which of them GPUs can be used?"
objectives:
- "Review one example of a code written with several methods of parallelization"
keypoints:
- "The serial code only allows one core to follow the instructions in the code"
- "OpenMP uses several cores on the same machine as all the cores are able to address the same memory"
- "OpenACC is capable to use both multiple cores and accelerators such as GPUs"
- "MPI is used for distributed parallel computing. Being able to use cores on different compute nodes."
- "OpenCL is an effort for a platform independent model for hybrid computing"
- "CUDA is the model used by NVIDIA for accessing compute power from NVIDIA GPUs"
---

We will pursue and exploration of several paradigms in parallel computing, from purely CPU computing to GPU computing. The paradigms chosen were OpenMP, OpenACC, OpenCL, MPI and CUDA. By using the same problem as baseline we hope to give a sense of perspective of how those different alternatives of parallelization work in practice.

We will be solving a very simple problem. The DAXPY function is the name given in BLAS to a routine that implements the function Y = A * X + Y where X and Y may be either native double-precision valued matrices or numeric vectors, and A is a scalar. We will be computing a trigonometric identity using DAXPY as the function that will be target of parallelization.

We will compute this formula:

<img src="https://render.githubusercontent.com/render/math?math=cos(2x) = cos^2(x)-sin^2(x)">

**The Cauchy-Schwarz Inequality**

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

**The Cauchy-Schwarz Inequality**

```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```

    <p id="This_Is_What_I_Want"> $$ (a-b)^2 $$</p>
    <p id="First_Input"> <input id="Value_A"></p>
    <p id="Second_Input"> <input id="Value_B"></p>
    <p id="Output"></p>
    <p id="Activate"><button onclick="RUN()">Test This out</button></p>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS_HTML,http://myserver.com/MathJax/config/local/local.js">
        function RUN() {
            var a = document.getElementById("Value_A").value
            var b = document.getElementById("Value_B").value
            document.getElementById("Output").innerHTML = "$$ (" + a + "-" + b + ")^2 $$";
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    </script>

For an array of vectors in a domain in the range of <img src="https://render.githubusercontent.com/render/math?math=x=[0:\pi]">

All the codes in this lesson will be compiled with the NVIDIA HPC compilers. That will give us some uniformity on the compiler choose as this compiler support several of the parallel models that will be demostrated.

To access the compilers on Thorny you need to load the module:

~~~
$> module load lang/nvidia/nvhpc/20.7
~~~
{: .language-bash}

To check that the compiler is available execute:

~~~
$> nvc --version
~~~
{: .language-bash}

You should get something like:

~~~
nvc 20.7-0 LLVM 64-bit target on x86-64 Linux -tp skylake
NVIDIA Compilers and Tools
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
~~~
{: .output}


## Serial Code

We will start without parallelization. The term given to codes that uses one single sequence of instructions is *serial*. We will use this to explain a few of the basic elements of C programming for those not familiar with the language.

The C programming language uses plain texts files to describe the program. The code needs to be compiled. Compiling means that using a software package called *compiler* the text file is interpreted resulting in a new file that contains the instructions that the computer can follow directly to run the program.

The program below shows a code that populates two vectors and computes a new vector being the subtraction of the input vectors.
Here is the code:

~~~
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
~~~
{: .language-c}

Most of this code will be seen in the other implementations, for those unfamiliar with C programming, the code will be reviewed in some detail. That should facilitate understanding when moving into the parallel versions of this code.

The first lines in the code are:

~~~
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VECTOR_SIZE 100
~~~
{: .language-c}

The first three indicate which files must be processed to access functions that will appear in the code below. In particular we need `stdio.h` because we are calling the function `printf`. We need `stdlib.h` for using the functions to allocate `malloc` and deallocate `free` arrays in memory. Finally we need `math.h` because we are calling math functions such as `sin` and `cos`.

The next line is a preprocessor instruction. Any place in the code where the word `VECTOR_SIZE` appears, excluding string constants, will be replace by the characters `100` before sending the code to the actual compiler. For us that help us to have a single place to change the code in case we want a larger or shorter array. As it is the lenght of the array is hardcoded meaning that in order to change the lenght of the arrays the source needs to be changed and recompiled.

The next lines are:

~~~
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
~~~
{: .language-c}

The first line in this chunk, indicates the main function on any program written in C. The code starts execution exactly starting on that line. A program written in C can read command line arguments from the shell and those arguments are stored in the array variable `argv` and the number of arguments with the integer `argc`

Next lines are declarations of the variables and the kind of each variable. Different from higher level languages like Python or R, in C each variable must have a very definite kind. `int` for integers, `double` for floating point numbers, ie, truncated real numbers. `size_t` is a kind defined on the `stdlib.h` header used to declare the number of bytes to allocate on each array.

The final lines are declarations for the 4 arrays that we will use. One for the domain X, the array A to store $cos(x)^2$, the array B to store $sin(x)^2$ and the array C for storing the difference of those two arrays.

Next lines are the core of the program:

~~~
for(i=0;i<VECTOR_SIZE;i++){
    X[i]=M_PI*(double)(i+1)/VECTOR_SIZE;
    A[i]=B[i]=C[i]=0.0;
}

for(i=0;i<VECTOR_SIZE;i++){
    A[i]=cos(X[i])*cos(X[i]);
    B[i]=sin(X[i])*sin(X[i]);
    C[i]=alpha*A[i]-B[i];
}
~~~
{: .language-c}

There are two loops in this chunk of code. The counter is the integer `i` that starts on 0 and stops before the variable reaches `VECTOR_SIZE`. In C language, the index of vectors of length N start in 0 and end in N-1. After each cycle the variable is increased by one with the instructions `i++` a short for `i=i+1`.

The vector X is filled with numbers starting with 0 and ending with $\pi$. Notice the use of M_PI, a constant declared in `math.h` that provides a long precision numerical value for PI. The other arrays are zeroed. Not really necessary but done here to show how several variables can receive the same value.

The next loop is actually the main piece of code that will change the most when we explore the different models of parallel programming. In this case, we have a single loop that will compute A, compute B and subtract those values to compute C.

The final portion of the code, shows the results of the calculations.

~~~
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
~~~
{: .language-c}

Here we have a conditional, for small arrays we will print the entire set of arrays, X, A, B and C as 4 columns of text on the screen. For larger arrays only the last 10 elements are shown.

Notice how to indicate the format of the numbers that will be shown on screen.
The string `%9.5f` means that each number will have 9 characters to display with 5 of them being decimals.  The character `f` means that the content of an floating point number will be shown. Other indicators are `f` for floats, `d` for integers and `s` for strings.

The next lines deallocate the memory used for the 4 arrays. Every program in written in C should return an integer at the end, with 0 meaning a successful completion of the program. Any other return value can be interpreted as some sort of failure.

To compile the code execute:

~~~
$> nvc daxpy_serial.c
~~~
{: source}

Execute the code with:

~~~
$> ./a.out
~~~
{: source}


That concludes the first program. No parallelism here. The code will use just one core, no matter how big the array is and how many cores you have available on your computer. This is a serial code and can only execute one instruction at a time.

No we will use this first program to present a few popular models for parallel computing.

## OpenMP: Shared-memory parallel programming.

The OpenMP (Open Multi-Processing) API is a model for parallel computing that supports multi-platform shared-memory multiprocessing programming in C, C++, and Fortran, on many platforms, instruction-set architectures and operating systems, including Solaris, AIX, HP-UX, Linux, macOS, and Windows. Shared-memory multiprocessing means that it can parallelize operations on multiple cores that are able to address a single pool of memory.

Programming in OpenMP consists of a set of compiler directives, library routines, and environment variables that influence run-time behavior.
We will focus only on the C language but there are equivalent directives for C++ and Fortran.

Programming in OpenMP consists in adding a few directives in critical places of the code that we want to parallelize, it is very simple to use and requires minimal changes to the source code.

Consider the next code based on the serial code with OpenMP directives:

~~~
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#define VECTOR_SIZE 100

int main( int argc, char* argv[] )
{
    int i,id;
    double alpha=1.0;

    // Number of bytes to allocate for N doubles
    size_t bytes = VECTOR_SIZE*sizeof(double);

    // Allocate memory for arrays X, A, B, and C on host
    double *X = (double*)malloc(bytes);
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    #pragma omp parallel for shared(A,B) private(i,id)
    for(i=0;i<VECTOR_SIZE;i++){
        id = omp_get_thread_num();
        printf("Initializing vectors id=%d working on i=%d\n", id,i);
        X[i]=M_PI*(double)(i+1)/VECTOR_SIZE;
        A[i]=cos(X[i])*cos(X[i]);
        B[i]=sin(X[i])*sin(X[i]);
    }

    #pragma omp parallel for shared(A,B,C) private(i,id) schedule(static,10)
    for(i=0;i<VECTOR_SIZE;i++){
        id = omp_get_thread_num();
        printf("Computing C id=%d working on i=%d\n", id,i);
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
~~~
{: .language-c}

The good thing about OpenMP is that not much of the code has to change in order to get a decent parallelization. As such, we will only focus of the  4 lines that have changed here:

~~~
#include <omp.h>
~~~
{: .language-c}

There is no need of importing any header for using OpenMP, however, we will use
function call to `omp_get_thread_num()` that is in the header above.

The next two sections with OpenMP directives are on top of the two loops, the initialization loop:

~~~
#pragma omp parallel for shared(A,B) private(i,id)
for(i=0;i<VECTOR_SIZE;i++){
    id = omp_get_thread_num();
~~~
{: .language-c}

And the evaluation of the vector:

~~~
#pragma omp parallel for shared(A,B,C) private(i,id) schedule(static,10)
for(i=0;i<VECTOR_SIZE;i++){
    id = omp_get_thread_num();
~~~
{: .language-c}

These two `#pragma` lines are sit on top of the for loops. From the point of view of the C language, they are just comments, so the language itself is not concern with them. A C compiler that support OpenMP will interpret those lines and will parallelize the for loops. The parallelization is posible because each evaluation of the `i` element is independent from the `j` element. That independence allows for different evaluations go to different cores.

There are several directives in the OpenMP specification. The directive `omp parallel for`, that is specific to parallelize for loops. There are others for assign executions to a core. For the parallel for directive there are arguments, one important set is the `shared` and `private` arguments that declare which variables will be shared on all the threads created by OpenMP and for which variables a copy will be created independent for each thread. The index `i` and the thread number `id` are always private. In this example, the variables are being declared explicitly even if that is not always needed.

The final argument is `schedule(static,10)` when the indices will be assign in chunks of 10.

OpenMP is a good option for easy parallelization of codes, but before OpenMP 4.0 the model was restricted only to parallelization with CPUs.

## MPI: Distributed Parallel Programming

Message Passing Interface (MPI) is a communication protocol for programming parallel computers. It is designed to be allow coordination of multiple cores on machines when cores are potentially independent, such as HPC clusters.

With MPI point-to-point and collective communication are supported.
Point to point communication is when one process send a message to another specific process, different processes are differentiated by their rank, an integer that uniquely identify the process.

MPI's goals are high performance, scalability, and portability. MPI remains the dominant model used in high-performance computing today.

The code below solves is a rewrite of the original serial program using point to point functions to distribute the computation across several MPI processes.

~~~
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
~~~
{: .language-c}

There are many changes in the code here, writing MPI code means important changes in the overall structure of the code as a result of organizing the sending and receiving of data from different processes.

The first change is the header:

~~~
#include "mpi.h"
~~~
{: .language-c}

All MPI programs must import that header to access the functions in the MPI API.

The next relevant chunk of code is:

~~~
MPI_Status status;

// Initialization of MPI environment
MPI_Init (&argc, &argv);

MPI_Comm_size (MPI_COMM_WORLD, &total_proc); //The total number of processes running in parallel
MPI_Comm_rank (MPI_COMM_WORLD, &rank); //The rank of the current process
~~~
{: .language-c}

In here we see a call to `MPI_Init()` the very first MPI function that must be call before any other. A variable `MPI_Status` is added to be used later for the receiving function. The call to `MPI_Comm_size` and `MPI_Comm_rank` will retrieve the total number of processes involved in the calculation, a number that could change at runtime. Each individual process receives a number called `rank` and we are storing the integer in the variable with that name.

In MPI we can avoid allocating big arrays full size in memory, but just allocating the portion of the array that will be used on each rank we can decrease the overall memory usage. A poorly written code will allocate entire arrays and used just a portion of the values. In here we are just allocating the amount of data actually needed.

~~~
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
~~~
{: .language-c}

The arrays, `ap`, `bp`, `cp` and `x` are smaller with size `n_per_proc` instead of `VECTOR_SIZE`. Notice that x must be initialized correctly with

~~~
    x[i]=M_PI*(rank*n_per_proc+i)/VECTOR_SIZE;
~~~
{: .language-c}

Each rank is just allocating and initializing the portion of data that it will manage.

The last chunk of source:

~~~
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
~~~
{: .language-c}

The MASTER rank (usually 0) will receive the data from the other ranks, and it could be used to assemble the complete array or simply to continue the execution based on the data processed by all the ranks. In this simple example we are just printing the data. The important lines here are:

~~~
MPI_Bsend(cp, n_per_proc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
~~~
{: .language-c}

and

~~~
MPI_Recv(buf, n_per_proc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
~~~
{: .language-c}

They send and receive the arrays in the first argument, in our case each array contains `n_per_proc` numbers of `MPI_DOUBLE` kind, (double in C), from the rank `i` with tag `0`, ranks that belong to `MPI_COMM_WORLD` the MPI world initialized.

The final MPI call of any program is:

~~~
MPI_Finalize();
//Terminate MPI Environment
return 0;
~~~
{: .language-c}

The calls `MPI_Init()` and `MPI_Finalize()` are the first and last MPI functions called on any program using MPI.

# OpenCL: A model for heterogeneous parallel computing

OpenCL is a framework for writing programs that execute across heterogeneous platforms consisting of CPUs, GPUs, digital signal processors, field-programmable gate arrays and other processors or hardware accelerators. OpenCL specifies programming languages for programming these devices and application programming interfaces to control the platform and execute programs on the compute devices.

Our program writen in OpenCL looks like this:

~~~
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#define VECTOR_SIZE 1024

//OpenCL kernel which is run for every work item created.
const char *daxpy_kernel =
"__kernel                                   \n"
"void daxpy_kernel(double alpha,     \n"
"                  __global double *A,       \n"
"                  __global double *B,       \n"
"                  __global double *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] - B[index]; \n"
"}                                          \n";

int main(void) {
  int i;
  // Allocate space for vectors A, B and C
  double alpha = 1.0;
  double *X = (double*)malloc(sizeof(double)*VECTOR_SIZE);
  double *A = (double*)malloc(sizeof(double)*VECTOR_SIZE);
  double *B = (double*)malloc(sizeof(double)*VECTOR_SIZE);
  double *C = (double*)malloc(sizeof(double)*VECTOR_SIZE);
  for(i = 0; i < VECTOR_SIZE; i++)
  {
    X[i] = M_PI*(double)(i+1)/VECTOR_SIZE;
    A[i] = cos(X[i])*cos(X[i]);
    B[i] = sin(X[i])*sin(X[i]);
    C[i] = 0;
  }

  // Get platform and device information
  cl_platform_id * platforms = NULL;
  cl_uint     num_platforms;
  //Set up the Platform
  cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id *)
  malloc(sizeof(cl_platform_id)*num_platforms);
  clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

  //Get the devices list and choose the device you want to run on
  cl_device_id     *device_list = NULL;
  cl_uint           num_devices;

  clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
  device_list = (cl_device_id *)
  malloc(sizeof(cl_device_id)*num_devices);
  clStatus = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

  // Create one OpenCL context for each device in the platform
  cl_context context;
  context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

  // Create memory buffers on the device for each vector
  cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(double), NULL, &clStatus);
  cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(double), NULL, &clStatus);
  cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,VECTOR_SIZE * sizeof(double), NULL, &clStatus);

  // Copy the Buffer A and B to the device
  clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(double), A, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(double), B, 0, NULL, NULL);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&daxpy_kernel, NULL, &clStatus);

  // Build the program
  clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "daxpy_kernel", &clStatus);

  // Set the arguments of the kernel
  clStatus = clSetKernelArg(kernel, 0, sizeof(double), (void *)&alpha);
  clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
  clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
  clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);

  // Execute the OpenCL kernel on the list
  size_t global_size = VECTOR_SIZE; // Process the entire lists
  size_t local_size = 64;           // Process one item at a time
  clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

  // Read the cl memory C_clmem on device to the host variable C
  clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(double), C, 0, NULL, NULL);

  // Clean up and wait for all the comands to complete.
  clStatus = clFlush(command_queue);
  clStatus = clFinish(command_queue);

  // Display the result to the screen
  for(i = 0; i < VECTOR_SIZE; i++)
    printf("%f * %f - %f = %f\n", alpha, A[i], B[i], C[i]);

  // Finally release all OpenCL allocated objects and host buffers.
  clStatus = clReleaseKernel(kernel);
  clStatus = clReleaseProgram(program);
  clStatus = clReleaseMemObject(A_clmem);
  clStatus = clReleaseMemObject(B_clmem);
  clStatus = clReleaseMemObject(C_clmem);
  clStatus = clReleaseCommandQueue(command_queue);
  clStatus = clReleaseContext(context);
  free(A);
  free(B);
  free(C);
  free(platforms);
  free(device_list);
  return 0;
}
~~~
{: .language-c}

There are many new lines here, most of it boilerplate code, so we will digest the most relevant chunks.

~~~
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
~~~
{: .language-c}

These lines are needed for reading the header for OpenCL, the location varies between MacOS and Linux, so preprocessor conditionals are used.

~~~
//OpenCL kernel which is run for every work item created.
const char *daxpy_kernel =
"__kernel                                   \n"
"void daxpy_kernel(double alpha,     \n"
"                  __global double *A,       \n"
"                  __global double *B,       \n"
"                  __global double *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] - B[index]; \n"
"}                                          \n";
~~~
{: .language-c}

Central to OpenCL is the idea of kernel function, the portion of code that will be offloaded to the GPU or any other accelerator. For academic purposes it is here written as a constant, but it can be a separate file for the kernel.
In this function we are just doing the difference alpha*A[i]-B[i]

~~~
// Get platform and device information
cl_platform_id * platforms = NULL;
cl_uint     num_platforms;
//Set up the Platform
cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
platforms = (cl_platform_id *)
malloc(sizeof(cl_platform_id)*num_platforms);
clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

//Get the devices list and choose the device you want to run on
cl_device_id     *device_list = NULL;
cl_uint           num_devices;

clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
device_list = (cl_device_id *)
malloc(sizeof(cl_device_id)*num_devices);
clStatus = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

// Create one OpenCL context for each device in the platform
cl_context context;
context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

// Create a command queue
cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

// Create memory buffers on the device for each vector
cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(double), NULL, &clStatus);
cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(double), NULL, &clStatus);
cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,VECTOR_SIZE * sizeof(double), NULL, &clStatus);
~~~
{: .language-c}

Most of this is just boilerplate code, it identifies the device, create an OpenCL context and allocate the memory on the device.

~~~
// Copy the Buffer A and B to the device
clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(double), A, 0, NULL, NULL);
clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(double), B, 0, NULL, NULL);

// Create a program from the kernel source
cl_program program = clCreateProgramWithSource(context, 1,(const char **)&daxpy_kernel, NULL, &clStatus);

// Build the program
clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

// Create the OpenCL kernel
cl_kernel kernel = clCreateKernel(program, "daxpy_kernel", &clStatus);

// Set the arguments of the kernel
clStatus = clSetKernelArg(kernel, 0, sizeof(double), (void *)&alpha);
clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);

// Execute the OpenCL kernel on the list
size_t global_size = VECTOR_SIZE; // Process the entire lists
size_t local_size = 64;           // Process one item at a time
clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
~~~
{: .language-c}

This section of the code is will be very similar when we explore CUDA, the main language in the next lectures. Here we are copying data from the host to the device, creating the kernel program and preparing the execution on the device.

~~~
// Read the cl memory C_clmem on device to the host variable C
clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(double), C, 0, NULL, NULL);

// Clean up and wait for all the comands to complete.
clStatus = clFlush(command_queue);
clStatus = clFinish(command_queue);

// Display the result to the screen
for(i = 0; i < VECTOR_SIZE; i++)
  printf("%f * %f - %f = %f\n", alpha, A[i], B[i], C[i]);

// Finally release all OpenCL allocated objects and host buffers.
clStatus = clReleaseKernel(kernel);
clStatus = clReleaseProgram(program);
clStatus = clReleaseMemObject(A_clmem);
clStatus = clReleaseMemObject(B_clmem);
clStatus = clReleaseMemObject(C_clmem);
clStatus = clReleaseCommandQueue(command_queue);
clStatus = clReleaseContext(context);
free(A);
free(B);
free(C);
free(platforms);
free(device_list);
return 0;
}
~~~
{: .language-c}

The final section clean the memory from the device, print the values of the arrays as we did on the serial code and free the memory on the host.

## OpenACC: Model for heterogeneous parallel programming

OpenACC (for open accelerators) is a programming standard for parallel computing developed by Cray, CAPS, Nvidia and PGI. The standard is designed to simplify parallel programming of heterogeneous CPU/GPU systems.

OpenACC is similar to OpenMP. In OpenMP, the programmer can annotate C, C++ and Fortran source code to identify the areas that should be accelerated using compiler directives and additional functions. Like OpenMP 4.0 and newer, OpenACC can target both the CPU and GPU architectures and launch computational code on them.

~~~
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
~~~
{: .language-c}

OpenACC works very similar to OpenMP, you add `#pragma` lines that are comments from the point of view of the C language but interpreted by the compiler if support for OpenACC is granted on the compiler.

On this chunk:

~~~
// sum component wise and save result into vector c
#pragma acc kernels copyin(A[0:VECTOR_SIZE],B[0:VECTOR_SIZE]), copyout(C[0:VECTOR_SIZE])
for(i=0; i<VECTOR_SIZE; i++) {
    C[i] = alpha*A[i]-B[i];
}
~~~
{: .language-c}

How OpenACC works is better understood with a more deep understanding of CUDA. The `#pragma` is parallelizing the for loop copying the A and B arrays to the memory on the GPU and returning the C array back. The main difference between OpenMP and OpenACC is for variables being able to operate on accelerators like GPUs those arrays and variables must be copied to the device and the results copied back to host.

## CUDA: Parallel Computing on NVIDIA GPUs

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA to work on its GPUs. The purpose of CUDA is leverage general purpose computing on CUDA-enabled graphics processing unit (GPU) sometimes termed GPGPU (General-Purpose computing on Graphics Processing Units).

The CUDA platform is a software layer that gives direct access to the GPU's virtual instruction set and parallel computational elements, for the execution of compute kernels. CUDA works as a superset of the C language and as such it cannot be compiled on a normal C language compiler. CUDA provides what is called the CUDA Toolkit a set of tools to compile and run programs written in CUDA.

~~~
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
~~~
{: .language-c}

There are two main sections in this code that depart from the C language:

~~~
// Kernel
__global__ void add_vectors(double alpha, double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < VECTOR_SIZE) c[id] = a[id] - b[id];
}
~~~
{: .language-c}

The function `add_vectors` will operate on the GPU using memory allocated on the device. The integer `id` will be associated to the exact index in the array based on the indices of thread and block the main components in the thread hierarchy of execution in CUDA.

~~~
// Allocate memory for arrays d_A, d_B, and d_C on device
double *d_A, *d_B, *d_C;
cudaMalloc(&d_A, bytes);
cudaMalloc(&d_B, bytes);
cudaMalloc(&d_C, bytes);
~~~
{: .language-c}

On this section the memory on the device is allocated. Those allocations are different from the arrays on the host and next lines will transfer data from the host to the device:

~~~
// Copy data from host arrays A and B to device arrays d_A and d_B
cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
~~~
{: .language-c}

The operation of transfer data from the host to the device is a blocking operation. The execution will not return to the CPU until the transfer is completed.

~~~
// Set execution configuration parameters
//      thr_per_blk: number of CUDA threads per grid block
//      blk_in_grid: number of blocks in grid
int thr_per_blk = 32;
int blk_in_grid = ceil( double(VECTOR_SIZE) / thr_per_blk );

// Launch kernel
add_vectors<<< blk_in_grid, thr_per_blk >>>(alpha, d_A, d_B, d_C);

// Copy data from device array d_C to host array C
cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
~~~
{: .language-c}

On this section we decide two important numbers in CUDA executionn that affect performance. The number of threads per block and the number of blocks in the grid. CUDA organizes parallel executions in threads those threads are grouped in a 3D arrays called blocks and blocks are arranged in a 3D array called a grid. For the simple case above we have just unidimensional blocks and unidimensional grids.

The call to `add_vectors` is a non-blocking operation. Execution return to the host as soon as the function is called. The device will work on that function independent from the host.

Only the function `cudaMemcpy()` will impose a barrier, waiting for the device to complete execution before copying the data back from device to host.

On the next lessons we will focus on CUDA to learn how to program GPUs and finally OpenACC a simpler model for fast offload of computation to the GPUs as we saw before.

{% include links.md %}
