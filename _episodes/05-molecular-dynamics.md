---
title: "Molecular Dynamics"
start: 600
teaching: 60
exercises: 0
questions:
- "What is Classical Molecular Dynamics (CMD)?"
- "How can we simulate the motion of particles subject to inter-particle forces?"
- "Which problems can be solved with CMD?"
- "How we can use GPUs to accelerate a CMD simulation"
objectives:
- "Learn the basics of Molecular Dynamics"
- "Learn how to use several CMD packages that can use GPUs to accelerate simulations"
- "We can solve Newton's equations of motion for relative motion of atoms subject to intramolecular forces."
keypoints:
- "We can use several methods to obtain a potential energy surface, from which inter-particle forces can be used to simulate the movement of particles subject to external and the interaction between particles.
can be derived.  We can then perform molecular dynamics simulations to solve Newton's equation of motion for the relative motion of atoms subject to realistic intramolecular forces."
- "Molecular Dynamics is an important tool for Chemistry and Engineering"
- "CMD uses force fields and Newton's Laws to simulate the evolution of a system of interacting particles"
- "Scientific packages that can use GPUs to accelerate simulations include Gromacs, LAMMPS, NAMD, and Amber"
---

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"> </script> <script src="https://unpkg.com/ngl@0.10.4/dist/ngl.js"></script>

Molecular Dynamics simulations provide a powerful tool for studying the motion of atomic, molecular, and nanoscale systems (including biomolecular systems).  In molecular dynamics simulation, the motion of the atoms is treated with classical mechanics; this means that the positions and velocities/momenta of the atoms are well-defined quantities. This in contrast with quantum mechanics where the theory sets limits to the ability to measure both quantities precisely. 

In classical molecular dynamics, positions and velocities are quantities updated using [Newton's laws of motion](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion) (or equivalent formulations).  

<!--
A brief overview of different applications of molecular dynamics simulations, as well as an introduction to the key working
equations can be found [here](https://github.com/FoleyLab/2019-Tapia-MolSSI/blob/master/Further_Reading/MD_MainPaper.pdf), with some additional details [here](https://github.com/FoleyLab/2019-Tapia-MolSSI/blob/master/Further_Reading/MD_TheoreticalBackground.pdf), as well as to references cited therein.  An even more concise summary of this exercise can be found in [these slides](https://github.com/FoleyLab/2019-Tapia-MolSSI/blob/master/Overview.pdf).
-->

The motion of the atoms over the duration of a molecular dynamics simulation is known as a trajectory.  The analysis of 
molecular dynamics trajectories can provide information about thermodynamic and kinetic quantities relevant to your system, and can also yield valuable insights into the mechanism of processes that occur over time.  Thus molecular dynamics simulations are employed to study an enormous range of chemical, material, and biological systems.  

One of the central quantities in a molecular dynamics simulation comes from the inter-particle forces.  From Newton's second law, we know that the acceleration a particle feels is related to the force it experiences;

$$ \vec{F} = m \vec{a}. $$

The acceleration of each particle in turn determines how the position and momentum of each particle change.  Therefore, the trajectories one observes in a molecular dynamics simulation are governed by the forces experienced by the particles being simulated.

 We will derive this force from the potential curve as follows,

$$ F(r) = -\frac{d}{dr} V(r), $$

where $$V(r)$$ denotes the potential as a function of the bond length that can be readily computed.


The potential function can be computed directly from quantum mechanical calculations, which we call those _ab_ _initio_ calculations. The alternative is to use approximations of the values we will get otherwise will be computed from running quantum mechanical equations. We call these approximations "force fields" and the resulting equations are called force-field molecular dynamics or Classical Molecular Dynamics (CMD).

There is a variety of computer algorithms available that permit Newton's Law to be solved for
many thousands of atoms even with modest computational resources. 
A classical algorithm for propagating particles under Newton's laws is called 
the Velocity Verlet algorithm.

This algorithm updates the positions and velocities of the particles once the
forces have been computed. Using the Velocity Verlet algorithm, the position of the $$i^{th}$$
particle is updated from time $$t$$ to time $$t+dt$$ according to equation:

$$ \vec{r}_i (t+dt) = \vec{r}_i (t) + \vec{v}_i(t) dt + \frac{1}{2} \vec{a}_i (t) dt^2$$

where $$\vec{r}_i(t)$$ represents the position vector of the $$i^{th}$$ particle at time $$t$$, $$\vec{v}_i(t)$$ 
represents the velocity vector of the $$i^{th}$$ particle at time $$t$$, and, $$\vec{a}_i(t)$$ represents the acceleration vector of the $$i^{th}$$ particle at time $$t$$. A MD code could use a Cartesian coordinate system so $$r$$ has
an $$x$$, $$y$$, and $$z$$ component: $$\vec{r} = r_x \hat{x} + r_y \hat{y} + r_z \hat{z}$$ and similarly for 
$$\vec{r}$$ and $$\vec{a}$$. 
In the previous expression, $$\hat{x}$$ is the unit vector along the $$x$$-axis. The velocity of the $$t^{th}$$ 
particle is updated from time $$t$$ to time $$t+dt$$ according to

$$ \vec{v}_i (t+dt) = \vec{v}_i(t) + \frac{1}{2}\left( \vec{a}_i (t) + \vec{a}_i (t+dt) \right) dt$$

and the acceleration comes from Newton's $$2^{nd}$$ law, $$\vec{a}_i = \frac{\vec{F_i}}{m_i}$$. 
Note that in the equation above, $$\vec{a}_i (t)$$
indicates the accelerations before the particle positions are updated and
$$\vec{a}_i (t+dt)$$ indicates the accelerations after the particle positions are updated. 

Hence, the velocity Verlet algorithm requires two force calculations and two
calculations of the accelerations for each iteration in the simulation. 

Practical application of the Velocity Verlet algorithm requires the specification of an
initial position and initial velocity for each particle. In this work, the initial positions are
specified by arranging the atoms in a simple cubic lattice, and the initial velocities are assigned
based on the Maxwell-Boltzmann distribution corresponding to a user-supplied initial
temperature.

A key ingredient in a molecular dynamics simulation is the potential energy function, also
called the force field. This is the mathematical model that describes the potential energy of the
simulated system as a function of the coordinates of its atoms. The potential energy function
should be capable of faithfully describing the attractive and repulsive interparticle interactions
relevant to the system of interest. For simulations of complex molecular or biomolecular
systems with many different atom types, the potential energy function may be quite
complicated, and its determination may itself be a subject of intense research. 

Because we are interested in simulating monatomic gasses,
we use a particularly simple potential energy function in our simulation known as the
Lennard-Jones potential, which was also used in Rahman’s original work,

$$U(r_{ij}) = 4 \epsilon \left( \left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^{6} \right)$$

The Lennard-Jones potential is defined by two parameters, $$\epsilon$$, and $$\sigma$$, which may be determined
by fitting simulations to experimental data, through *ab initio* calculations, or by a combination
of experiment and calculation. The parameter $$\epsilon$$ is related to the strength of the interparticle
interactions and has dimensions of energy. The value of $$\epsilon$$ manifests itself as the minimum
value of the potential energy between a pair of particles, $$min(U(r_{ij}))= -\epsilon$$. 

The parameter $$\sigma$$ is related to the effective size of the particles and has dimensions of length. 
In effect, the value of $$\sigma$$ determines at which separations attractive forces dominate and at which separations repulsions dominate. 

For example, the value of $$\sigma$$ determines the interparticle separation that minimizes the Lennard-
Jones potential by the relation $$r_{min} = 2^{\frac{1}{6}}\sigma$$ where $$U(r_{min}) = \epsilon$$.

In the case of Lennard-Jone's potential, only the scalar separation between a pair of particles is needed to determine the potential energy of the pair within the Lennard-Jones model. 

The total potential energy of the system of $$N$$ particles is simply the sum of the potential energy of all unique pairs. This potential neglects orientation effects which may be important for describing molecules that lack
spherical symmetry and also neglects polarization effects which may arise in many chemically
relevant situations, water for example. Lennard-Jones parameters for various noble gasses
are provided in the table below.

| Particle    | m (kg)                   | $$\sigma (m)$$            | $$\epsilon (J)$$         |
| ----------- | ------------------------ | ------------------------- |--------------------------|
| Helium      | $$6.646 \times 10^{27}$$ | $$2.64 \times 10^{10}$$   | $$1.50 \times 10^{22}$$  |
| Neon        | $$3.350 \times 10^{26}$$ | $$2.74 \times 10^{10}$$   | $$5.60 \times 10^{22}$$  |
| Argon       | $$6.633 \times 10^{26}$$ | $$3.36 \times 10^{10}$$   | $$1.96 \times 10^{21}$$  |
| Krypton     | $$1.391 \times 10^{26}$$ | $$3.58 \times 10^{10}$$   | $$2.75 \times 10^{21}$$  |
| Xeon        | $$2.180 \times 10^{26}$$ | $$3.80 \times 10^{10}$$   | $$3.87 \times 10^{21}$$  |

Once the potential energy function has been specified, the forces on the particles may be
calculated from the derivative on the potential energy function with respect to interparticle
separation vectors:


$$\vec{𝐹}_i = − \sum_{j\neq i}^{N} \frac{\partial U(\vec{r}_{ij})}{\partial \vec{r}_{ij}}$$

That is, each particle experiences a unique force from each of the remaining particles in the
system. Each unique force is related to the derivative of the potential energy with respect to
the separation vector of the particle pair,

$$\vec{F}_{ij}(\vec{r}_{ij}) = \frac{24\epsilon}{\sigma^2} \vec{r}_{ij} \left( 2\left( \frac{\sigma}{\vec{r}_{ij}} \right)^{14} - \left( \frac{\sigma}{\vec{r}_{ij}} \right)^{8} \right)$$

Note that, unlike the potential energy, the pair force is fundamentally a vector quantity and has
a direction associated with it. We note that the two $$\frac{\sigma}{\vec{r}_{ij}}$$ 
terms in the force can be equivalently evaluated in terms of the scalar separation, $$\vec{r}_{ij}$$, 
because they contain even powers of the separation vector, and even powers of the separation 
vector are equivalent to the same even power of the scalar separation. 

Because the Lennard-Jones potential has a minimum at the separation $$r_{min} = 2^{\frac{1}{6}}\sigma$$, 
the force goes to zero at separation $$r_{min}$$. 

Once all the unique pair force vectors $$\vec{F}_{ij}$$ are determined from evaluation of based on the
coordinates of the system, the total force vector $$\vec{F}_i$$ acting on the $$i^{th}$$ particle is computed as the
sum over all the unique pair forces. Hence, the potential energy function is a key ingredient in
determining the dynamics through its impact on the forces, and
therefore, the instantaneous position and velocity. 

The instantaneous positions and velocities of the particles constitute the raw
data generated by a molecular dynamics simulation. These trajectories may be directly
visualized, and a variety of physical quantities can be derived from information 
contained in these trajectories.

## LAMMPS

LAMMPS is a C++ code for classical molecular dynamics with a focus on material
modeling. It is an acronym for Large-scale Atomic/Molecular Massively Parallel Simulator
It is also a free software optimized for OPENMP, OPENMPI and CUDA libraries. The code is
flexible in order to simulate various physical situations and can perform simulation with
many different algorithms. The price to pay is to write an input script file which defines
all parameters as well as the quantities to be monitored. 

There are many parameters that can be set on LAMMPS
All possibilities can be found in the software manual at [LAMMPS MANUAL](https://lammps.sandia.gov/doc/Manual.pdf). 
The last version of the manual (Release 15Jun2023) contains 2786 pages, which illustrates the complexity of the
software. To start with LAMMPS is not always easy, and if you are a beginner you must
follow the basic instructions and understand the meaning of the lines of a script file. The
goal of this session is to illustrate the versatility of the software and the quality of the
parallelization.


The binary file is called ``lmp``.
In order to have a brief description of LAMMPS. type ``lmp -help``. 
You will learn how the software has been compiled and all methods are available. 
If some options are missing, you can download the source tarball and compile the software with the options you can
use with your computer. See the appendix for an installation.

You can start a simulation by using the command

	$> lmp -in in.system

where ``in.system`` is a script file defining the system (particle types, ensemble, forces,
boundary conditions,...), the different observables to be calculated, and different correlation
functions (static and dynamic).


### Lennard-Jones Liquid

We will use this first example of a LAMMPS input file

      # 3d Lennard-Jones melt
      
      variable	x index 1
      variable	y index 1
      variable	z index 1
      
      variable	xx equal 50*$x
      variable	yy equal 50*$y
      variable	zz equal 50*$z
      
      units		lj
      atom_style	atomic
      
      lattice		fcc 0.8442
      region		box block 0 ${xx} 0 ${yy} 0 ${zz}
      create_box	1 box
      create_atoms	1 box
      mass		1 1.0
      
      velocity	all create 1.44 87287 loop geom
      
      pair_style	lj/cut 2.5
      pair_coeff	1 1 1.0 1.0 2.5
      
      neighbor	0.3 bin
      neigh_modify	delay 0 every 20 check no
      
      fix		1 all nve
      
      run		300

This input will simulation an Atomic fluid of particles following Lennard-Jones potential.
In its original version the parameters are:

 * 32,000 atoms for 100 timesteps
 * reduced density = 0.8442 (liquid)
 * force cutoff = 2.5 sigma
 * neighbor skin = 0.3 sigma
 * neighbors/atom = 55 (within force cutoff)
 * NVE time integration 

We can explore the variables used here

#### ``units``

This command sets the style of units used for a simulation. It determines the units of all 
quantities specified in the input script and data file, as well as quantities output to the 
screen, log file, and dump files. Typically, this command is used at the very beginning of an 
input script.

We are running a unitless simulation for a Lennard-Jones potential. For style lj, all quantities 
are unitless.
Without loss of generality, LAMMPS sets the fundamental quantities mass, $$\sigma$$ , $$\epsilon$$, 
and the Boltzmann constant $$k_B = 1$$. 
The masses, distances, energies you specify are multiples of these fundamental values.

#### ``atom_style``

Define what style of atoms to use in a simulation. This determines what attributes are associated with the atoms.
This command must be used before a simulation is setup via a read_data, read_restart, or create_box command.

We are using ``atom_style	atomic`` because we are dealing with point-like particles.
The choice of style affects what quantities are stored by each atom, what quantities are 
communicated between processors to enable forces to be computed, and what quantities are 
listed in the data file read by the read_data command.

#### ``lattice``

A lattice is a set of points in space, determined by a unit cell with basis atoms, that is 
replicated infinitely in all dimensions. 

A lattice is used by LAMMPS in two ways. First, the ``create_atoms`` command creates atoms on the 
lattice points inside the simulation box. Note that the ``create_atoms`` command allows different 
atom types to be assigned to different basis atoms of the lattice.

In our case, we have particles under a reduced density of 0.8442. 

#### ``region``

This command defines a geometric region of space. Various other commands use regions. For example, the region
can be filled with atoms via the create_atoms command. Or a bounding box around the region, can be used to define
the simulation box via the create_box command.

The region is identified with the name "box" and for style block, the indices from 1 to 6 correspond 
to the xlo, xhi, ylo, yhi, zlo, zhi surfaces of the block. 

#### ``create_box``

This command creates a simulation box based on the specified region. Thus a region command must first be used to define a geometric domain. It also partitions the simulation box into a regular 3d grid of rectangular bricks, one per processor, based on the number of processors being used and the settings of the processors command. 

We will have just one box in the region that we defined.

#### ``create_atoms``

This command creates atoms (or molecules) within the simulation box, either on a lattice, or a single atom (or molecule), or on a surface defined by a triangulated mesh, or a random collection of atoms (or molecules). 

For the box style, the ``create_atoms`` command fills the entire simulation box with particles on the lattice.

#### ``mass``

Set the mass for all atoms of one or more atom types. Per-type mass values can also be set in the read_data data file using the “Masses” keyword. See the units command for what mass units to use.

We only have one type of atoms and the mass is unitless for lj 

#### ``velocity``

Set or change the velocities of a group of atoms in one of several styles. 
For each style, there are required arguments and optional keyword/value parameters. 

In our case we are associating random values of velocity for all the particles.
The create style generates an ensemble of velocities using a random number generator 
with the specified seed at the specified temperature.

#### ``pair_style``

Set the formula LAMMPS uses to compute pairwise interactions. 
The particles in our simulation box are all suject to pair-wise interactions with an energy 
cutoff of $$2.5 \sigma$$

#### ``pair_coeff``

Specify the pairwise force field coefficients for one or more pairs of atom types.


## How to run LAMMPS using GPUs

Submit a request for an interactive job:

    $> srun -p comm_gpu_inter -G 3 -t 2:00:00 -c 24 --pty bash

With this command, the partition will be ``-p comm_gpu_inter ``. We are requesting 3 GPUs (``-G 3``)
during 2 hours ``-t 2:00:00`` with 24 CPU cores ``-c 24`` and bash on the shell ``--pty bash``.

Once you get access to the compute node, load singularity.

    $> module load singularity

Check the GPUs that you received:

    $ nvidia-smi 
    Tue Aug  1 07:51:25 2023       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  Quadro RTX 6000                 Off| 00000000:12:00.0 Off |                    0 |
    | N/A   29C    P8               22W / 250W|      0MiB / 23040MiB |      0%   E. Process |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
    |   1  Quadro RTX 6000                 Off| 00000000:13:00.0 Off |                    0 |
    | N/A   28C    P8               24W / 250W|      0MiB / 23040MiB |      0%   E. Process |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
    |   2  Quadro RTX 6000                 Off| 00000000:48:00.0 Off |                    0 |
    | N/A   27C    P8               12W / 250W|      0MiB / 23040MiB |      0%   E. Process |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
                                                                                             
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+

One important piece of information about GPUs is the Compute Capability.

The compute capability of a device is represented by a version number, also sometimes called 
its “SM version”.
This version number identifies the features supported by the GPU hardware and is used by 
applications at runtime to determine which hardware features and/or instructions are available 
on the present GPU.

All the GPUs are Quadro RTX 6000. These GPUs have Compute Capability 7.5. See more about 
compute capabilities on the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) and the tables with the SM of various GPU cards on this 
[table](https://developer.nvidia.com/cuda-gpus#compute). 

Open a shell on the container:

    $> singularity shell --nv /shared/containers/NGC_LAMMPS_patch_3Nov2022.sif 
    Singularity>

The image ``NGC_LAMMPS_patch_3Nov2022.sif`` has been optimized by NVIDIA for running LAMMPS on GPUs.
There are several binaries inside depending on the Compute Capabilities of the GPU. 
We need to adjust the environment variables ``$PATH`` and ``$LD_LIBRARY_PATH`` according to the 
SM number of the card.

As the SM number for QUADRO RTX 6000 is 7.5 we can set those variables like this:

    Singularity> export PATH=/usr/local/lammps/sm75/bin:$PATH
    Singularity> export LD_LIBRARY_PATH=/usr/local/lammps/sm75/lib:$LD_LIBRARY_PATH

With this we are ready to launch LAMMPS. The binary is called ``lmp``. See for example the help 
message from the command, execute:

    Singularity> mpirun lmp -help

    Large-scale Atomic/Molecular Massively Parallel Simulator - 3 Nov 2022

    Usage example: lmp -var t 300 -echo screen -in in.alloy
    
    List of command line options supported by this LAMMPS executable:
    
    -echo none/screen/log/both  : echoing of input script (-e)
    -help                       : print this help message (-h)
    -in none/filename           : read input from file or stdin (default) (-i)
    -kokkos on/off ...          : turn KOKKOS mode on or off (-k)
    -log none/filename          : where to send log output (-l)
    -mdi '<mdi flags>'          : pass flags to the MolSSI Driver Interface
    -mpicolor color             : which exe in a multi-exe mpirun cmd (-m)
    -cite                       : select citation reminder style (-c)
    -nocite                     : disable citation reminder (-nc)
    -nonbuf                     : disable screen/logfile buffering (-nb)
    -package style ...          : invoke package command (-pk)
    -partition size1 size2 ...  : assign partition sizes (-p)
    -plog basename              : basename for partition logs (-pl)
    -pscreen basename           : basename for partition screens (-ps)
    -restart2data rfile dfile ... : convert restart to data file (-r2data)
    -restart2dump rfile dgroup dstyle dfile ... 
                                : convert restart to dump file (-r2dump)
    -reorder topology-specs     : processor reordering (-r)
    -screen none/filename       : where to send screen output (-sc)
    -skiprun                    : skip loops in run and minimize (-sr)
    -suffix gpu/intel/opt/omp   : style suffix to apply (-sf)
    -var varname value          : set index style variable (-v)
    
    OS: Linux "Ubuntu 20.04.5 LTS" 3.10.0-1160.24.1.el7.x86_64 x86_64
    
    Compiler: GNU C++ 10.3.0 with OpenMP 4.5
    C++ standard: C++14
    MPI v3.1: Open MPI v4.1.3rc2, package: Open MPI root@95f4f5de9494 Distribution, ident: 4.1.3rc2, repo rev: v4.1.3, Unreleased developer copy
    
    Accelerator configuration:
    
    KOKKOS package API: CUDA Serial
    KOKKOS package precision: double
    OPENMP package API: OpenMP
    OPENMP package precision: double

There are two variables we will use for executing LAMMPS. 
    
    Singularity> gpu_count=3                                                 
    Singularity> input=in.lj.txt 

Execute LAMMPS:

    Singularity> mpirun -n ${gpu_count} lmp -k on g ${gpu_count} -sf kk -pk kokkos cuda/aware on neigh full comm device binsize 2.8 -var x 8 -var y 4 -var z 8 -in ${input} 
    
On this command line we are running LAMMPS using MPI launching 3 MPI processes. 
The number of processes matches the number of GPUs ``mpirun -n ${gpu_count}``

To use GPUs LAMMPS relies on Kokkos.
Kokkos is a templated C++ library that provides abstractions to allow a single implementation of an application kernel (e.g. a pair style) to run efficiently on different kinds of hardware, such as GPUs, Intel Xeon Phis, or many-core CPUs. See more on [Kokkos documentation page](https://github.com/kokkos/kokkos)

Use the “-k” command-line switch to specify the number of GPUs per node. 
Typically the -np setting of the mpirun command should set the number of MPI tasks/node to be 
equal to the number of physical GPUs on the node. 

In our case we have ``-k on g ${gpu_count}``

The suffix to the style is kk for the particles to be accelerated with GPUs
Our simulation is creating more than 27 million atoms:

    LAMMPS (3 Nov 2022)           
    KOKKOS mode is enabled (src/KOKKOS/kokkos.cpp:106) 
      will use up to 3 GPU(s) per node       
      using 1 OpenMP thread(s) per MPI task    
    Lattice spacing in x,y,z = 1.6795962 1.6795962 1.6795962  
    Created orthogonal box = (0 0 0) to (403.10309 201.55154 403.10309)  
      1 by 1 by 3 MPI processor grid                                
    Created 27648000 atoms                                   
      using lattice units in orthogonal box = (0 0 0) to (403.10309 201.55154 403.10309) 
      create_atoms CPU = 3.062 seconds             

At the end of the simulation you get some stats about the performance

    Performance: 4270.015 tau/day, 9.884 timesteps/s, 273.281 Matom-step/s
    73.6% CPU use with 3 MPI tasks x 1 OpenMP threads
    
    MPI task timing breakdown:
    Section |  min time  |  avg time  |  max time  |%varavg| %total
    ---------------------------------------------------------------
    Pair    | 0.20305    | 0.20387    | 0.20487    |   0.2 |  0.67
    Neigh   | 4.7754     | 4.8134     | 4.837      |   1.2 | 15.86
    Comm    | 2.1789     | 2.2052     | 2.2234     |   1.3 |  7.27
    Output  | 0.00077292 | 0.021803   | 0.037289   |  10.4 |  0.07
    Modify  | 22.898     | 22.92      | 22.941     |   0.4 | 75.52
    Other   |            | 0.1868     |            |       |  0.62
    
    Nlocal:      9.216e+06 ave 9.21655e+06 max 9.21558e+06 min
    Histogram: 1 0 0 1 0 0 0 0 0 1
    Nghost:         811154 ave      811431 max      810677 min
    Histogram: 1 0 0 0 0 0 0 0 1 1
    Neighs:              0 ave           0 max           0 min
    Histogram: 3 0 0 0 0 0 0 0 0 0
    FullNghs:  6.90671e+08 ave 6.90734e+08 max 6.90632e+08 min
    Histogram: 1 1 0 0 0 0 0 0 0 1
    
    Total # of neighbors = 2.0720116e+09
    Ave neighs/atom = 74.94255
    Neighbor list builds = 15
    Dangerous builds not checked
    Total wall time: 0:00:39


Compare those to run the same simulation using a single GPU

    Performance: 1503.038 tau/day, 3.479 timesteps/s, 96.194 Matom-step/s
    73.4% CPU use with 1 MPI tasks x 1 OpenMP threads
    
    MPI task timing breakdown:
    Section |  min time  |  avg time  |  max time  |%varavg| %total
    ---------------------------------------------------------------
    Pair    | 2.6815     | 2.6815     | 2.6815     |   0.0 |  3.11
    Neigh   | 14.125     | 14.125     | 14.125     |   0.0 | 16.38
    Comm    | 0.73495    | 0.73495    | 0.73495    |   0.0 |  0.85
    Output  | 0.0019922  | 0.0019922  | 0.0019922  |   0.0 |  0.00
    Modify  | 68.193     | 68.193     | 68.193     |   0.0 | 79.09
    Other   |            | 0.4887     |            |       |  0.57
    
    Nlocal:     2.7648e+07 ave  2.7648e+07 max  2.7648e+07 min
    Histogram: 1 0 0 0 0 0 0 0 0 0
    Nghost:    1.60902e+06 ave 1.60902e+06 max 1.60902e+06 min
    Histogram: 1 0 0 0 0 0 0 0 0 0
    Neighs:              0 ave           0 max           0 min
    Histogram: 1 0 0 0 0 0 0 0 0 0
    FullNghs:  2.07201e+09 ave 2.07201e+09 max 2.07201e+09 min
    Histogram: 1 0 0 0 0 0 0 0 0 0
    
    Total # of neighbors = 2.0720116e+09
    Ave neighs/atom = 74.94255
    Neighbor list builds = 15
    Dangerous builds not checked
    Total wall time: 0:02:00

And compare with a CPU only execution asking for 3 CPU cores. The simple command for running 
without GPUs will be:

    Singularity> mpirun -n 3 lmp -var x 8 -var y 4 -var z 8 -in ${input}  

And the results:

    Performance: 87.807 tau/day, 0.203 timesteps/s, 5.620 Matom-step/s
    96.4% CPU use with 3 MPI tasks x 1 OpenMP threads
    
    MPI task timing breakdown:
    Section |  min time  |  avg time  |  max time  |%varavg| %total
    ---------------------------------------------------------------
    Pair    | 1109.8     | 1163.9     | 1232.7     | 150.1 | 78.86
    Neigh   | 161.57     | 164.1      | 167.77     |  20.7 | 11.12
    Comm    | 13.422     | 94.55      | 155.54     | 614.4 |  6.41
    Output  | 0.03208    | 0.040559   | 0.045395   |   3.0 |  0.00
    Modify  | 38.403     | 41.45      | 47.511     |  66.6 |  2.81
    Other   |            | 11.91      |            |       |  0.81
    
    Nlocal:      9.216e+06 ave 9.21655e+06 max 9.21558e+06 min
    Histogram: 1 0 0 1 0 0 0 0 0 1
    Nghost:         811154 ave      811431 max      810677 min
    Histogram: 1 0 0 0 0 0 0 0 1 1
    Neighs:    3.45335e+08 ave 3.45582e+08 max 3.45106e+08 min
    Histogram: 1 0 0 0 1 0 0 0 0 1
    
    Total # of neighbors = 1.0360058e+09
    Ave neighs/atom = 37.471275
    Neighbor list builds = 15
    Dangerous builds not checked
    Total wall time: 0:24:49

Not using the GPUs makes the simulation take 24 minutes. 
The 3 GPUs lower the time of simulation to just 40 seconds


{% include links.md %}
