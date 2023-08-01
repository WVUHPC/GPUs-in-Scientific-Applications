---
title: "Other Applications"
teaching: 60
exercises: 0
questions:
- "Beyond DL and CMD, which other applications can use accelerators?"
- "What is Computational Fluid Dynamics (CFD)?"
- "How to use GPUs to accelerate CFD simulations?"
objectives:
- "Learn how to use GPUs to accelerate other scientific applications"
keypoints:
- "GPUs can be used to accelerate other applications such as CFD, signal processing and Data Analysis"
---

# Parallelization

While GPUs do have the potential to dramatically speed up certain computational tasks, they are not a cure-all. In order to predict whether or not a certain problem can benefit from the use of these accelerators, we must determine whether or not it is *parallelizable*. In essence, we must ask if we can break up parts of the problem into smaller, *independent* tasks. It is critical that these small tasks depend on each other as little as possible; otherwise, the extra time taken to communicate between the processors will compromise any gains we might have made. 

For a simple, non-computational example of a parallelizable problem, suppose we are selling tickets to a theme park. These transactions are independent of each other; that is, the tickets sold before the current one have no affect on how the current transaction is done. As such, we can freely hire as many people as we want to sell tickets more efficiently. 

For an example of something that is *not* parallelizable, suppose that we wanted to fill an array with the first hundred Fibonacci numbers, where the $n^{\rm th}$ Fibonacci number $F_{n+2} = F_{n+1} + F_n$ by definition, and $F_0 = 0$ and $F_1 = 1$. Using python for simplicity, a reasonable algorithm to do this would be as follows:

```python
import numpy as np

N = 100                 # Set the length of the array
Fib_out = np.zeros(100) # Instantiate a list of only zeros
Fib_out[0] = 0          # Set initial values
Fib_out[1] = 1

for i in range(N-2):
    Fib_out[i+2] = Fib_out[i+1] + Fib_out[i]
```

At first glance, we might hope that we could paralellize the for-loop. However, notice that whenever we compute the next number in the sequence, we must reference other cells in the array. That is, to set the last element of the array, we must have set the second-last and third-last elements, and so on for each value. So, if we tried to assign each iteration of the loop to a different processor on a GPU, each one would either have to wait for the previous iterations to finish anyways, or else give a nonsensical answer. In cases like these, we need to either try to rewrite the code to minimize or eliminate these references to the results of other tasks, or else try to find other ways to speed up our code.

# Computational Fluid Dynamics

The most straightforward problems to parallelize are referred to as "embarrassingly parallel", where little-to-no work is needed to get the code ready to run in parallel to see significant increases in speed. When working with such problems, we'll likely find that the most straightforward way to implement our chosen algorithm is already good enough.

When implementing numerical methods to solve complicated partial differential equations, we often encounter such embarrassingly parallel algorithms. Partial differential equations (PDEs) can be used to describe many phenomena, especially in physics and engineering, where they can be found in everything from thermodynamics and quantum mechanics to electromagnetism and general relativity. Unfortunately, these equations can be incredibly difficult and even impossible to solve for all but the simplest systems. Thus, it is necessary to numerically solve these problems with computers. 

Generally, these equations describe a system that changes in time, and at any given time, *how* that system changes depends on the *state* of the system only at that time. When we solve the equations numerically to simulate the system, we have to discretize our domain by dividing it into small, finite regions. While there are many ways to do this, the end result is that we have large arrays of numbers that describe the state of our system at some time. We also have to discretize time, dividing it into small, finite timesteps. The state of the system at any given timestep depends only on its state at the previous timestep. Thus, *within* each timestep, we can see that the computation of the new values for each point are *completely* independent of each other, making problems like these relatively straightforward to run in parallel. 

Computational fluid dynamics is an example of this type of problem. Here, we are interested in numerically solving the Navier-Stokes equations, which govern how variables like the temperature, pressure, and velocity of liquids and gases change over time.

# Examples using Converge

## Example 1:


## Example 2: 


{% include links.md %}

