## Assignment 2: Lenia — An Artificial Life

**Authors:** Uroš Lotrič, Davor Sluga  
**Date:** March 2026

## Introduction

The [Lenia project](https://content.wolfram.com/sites/13/2019/10/28-3-1.pdf) started by experimenting with [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) variations. It is a generalisation of the Game of Life with continuous space, time, and states. As a consequence, it enables the generation of more complex autonomous creatures. The Game of Life and the Lenia are cellular automata. A cellular automaton is a grid of cells, each having a particular state at a moment. Cells are repeatedly updated according to a local rule, taking into account each cell and its neighbours.

In the Game of Life, the cells are arranged in a rectangular grid, time runs in discrete steps, and each cell has eight neighbouring cells (radius 1), which can take only discrete values 0 (dead) or 1 (alive). The new state of a cell is determined by its current state and the number of alive neighbouring cells.

In Lenia, the space is continuous, but for computer simulation purposes, it is again arranged in a rectangular grid. However, by creating smaller cells, the simulation becomes more accurate. Similarly, time is continuous but discretised for simulation purposes. The discrete time step can take any value; a smaller value results in a more detailed simulation. Next, the neighbourhood in Lenia is much broader. In our simulation, we will limit the radius to 13. Lastly, the state of a cell is represented by a continuous value bounded to the interval [0, 1]. Instead of simply counting alive neighbouring cells, the Lenia grid is convolved with a 2D ring kernel applied to a square matrix of size 26x26. The convolution result passes through a Gaussian-based growth function that determines cell development.


## Lenia simulation

The following listing shows the core simulation functions written in Python for brevity. You can find the whole Python script [here](src/lenia/python/lenia.py).

``` python
# Gaussian function
def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x-mu)/sigma)**2)

# Ring convolution kernel construction
def kernel_lenia(R, mu, sigma):
    y, x = np.ogrid[-R:R, -R:R]
    dist = np.sqrt((1+x)**2 + (1+y)**2) / R
    K = gauss(dist, mu, sigma)
    K[dist > 1] = 0
    K = K / np.sum(K)
    return K

# growth criteria
def growth_lenia(C, mu, sigma):
    return -1 + 2 * gauss(C, mu, sigma) 

# Lenia iteration t -> t+dt
def evolve_lenia(world, kernel, mu, sigma, dt):  
    # store result of convolution to a new matrix 
    C = sp.signal.convolve2d(world, kernel, mode='same', boundary='wrap') 
    # update the board 
    world = world + dt * growth_lenia(C, mu, sigma)
    # cell values should remain in interval [0, 1]
    world = np.clip(world, 0, 1)
    return world
```
The function `kernel_lenia` constructs a constant ring kernel. The core function of the simulation is `evolve_lenia`, which we call in every iteration. It first convolves the current state `world` with a kernel, then computes cell development by passing the convolution result `C` to the function `growth_lenia`. The visualisation of the growth function is shown below.

![The growth function.](img/growth.png)

Below, we can see the ring kernel and Orbium creature you can use in the simulation. If you place the orbium creature at the start into the world grid, it will keep its shape while it moves around as seen in the provided animation. For other exciting kernels and creatures, consult the [Lenia web page](https://chakazul.github.io/lenia.html).

![The kernel and the orbium](img/kernel-creature.png)

![Simulation.](img/lenia.gif)


## Parallel Lenia simulation

Although it is interesting to quest for new creatures and observe their development through time, this should not be your focus. The problem is also interesting from a parallelisation and code optimisation perspective. The evolution of Lenia cellular automata over time can be easily parallelised. In each iteration, each cell's state can be computed independently, and thus in parallel, based on the values of the neighbouring cells from the previous iteration. Of course, there exists a dependence between iterations, so all of the computations in the previous iteration need to finish before we proceed to the next iteration.


## Assignment

Implement a parallel version of the Lenia simulation in C/C++ and CUDA that evolves the initial world for a given number of iterations on the GPU and outputs the final grid state. You can start from the provided [sequential C code](src/lenia/), which already includes examples of build and run scripts to utilise GPUs on the Arnes cluster, as well as the initialisation code for the grid to produce moving orbiums. Note that the C code intentionally follows the Python script closely and is poorly optimised; feel free to improve it. It also includes optional code to generate animations, which you can use to examine the results. 

### Code organization
- `python/lenia.py` -> Python reference implementation.
- `Makefile` -> Project build rules.
- `run_lenia.sh` -> Sbatch script to acquire resources on the Arnes cluster, build and run the Lenia simulator.
- `src/`
    - `main.c` -> Main project file.
    - `lenia.cu` -> Lenia simulation code.
    - `orbium.c` -> Code for placement of [Orbium creatures](https://ar5iv.labs.arxiv.org/html/2005.03742/assets/fig3a1.png).
    - `gifenc.c` -> Code for generating gif animations; taken from [here](https://github.com/lecram/gifenc).


### **Basic tasks (for grades 6-8):**

- Parallelise the algorithm using CUDA as efficiently as possible. Look into lecture [stencil code samples](../../lectures/12-patterns/files/stencil/) for inspiration. 
- Avoid unnecessary memory transfers between the host and the device. When dividing the workload, find the optimal thread block size.
- Measure the execution time of the CUDA algorithm on Arnes cluster for different world sizes. Use the next grid sizes: 256x256, 512x512, 1024x1024, 2048x2048 and 4096x4096. Benchmark the algorithm on 100 simulation steps. When measuring time, the data transfers to and from the GPU must also be included. 
- Compute the speed-up $S=t_s/t_p$ of your algorithm for each image size; $t_s$ is the execution time of the sequential algorithm on the CPU, and $t_p$ is the execution time of the parallel algorithm on the GPU. Run the algorithm multiple times (at least 5) and average the measurements. Note that the base code for the largest grid takes more than an hour to run. You can do only one run in such a case.
- Visualise the resulting final state (don't put it in the report, but store it separately). You can even create an animation that shows how the world evolves over time. Do not include the time required to produce the animation in the time measurements or speedups.
- Write a short report (1-2 pages) summarising your solution and presenting the measurements performed on the cluster. The main focus should be on presenting and explaining the time measurements and speed-ups.
- Hand in your code and the report (one submission per pair) to ucilnica through the appropriate form by the specified deadline (**14. 4. 2026**) and defend your code and report during labs.

### **Bonus tasks (for grades 9-10):**
- Optimise and parallelise (with OpenMP) the provided sequential code. Use your improved code as the baseline (with the optimal number of cores) for the measurements when computing speedups against the GPU.
- Use shared memory within thread blocks to store local tiles of the world grid and other data structures that make sense, allowing for faster memory accesses. 
- Experiment with and optimise how the simulation grid is split between thread blocks, e.g. square blocks, stripes, etc.
- Utilise two GPUs to perform the simulation; split the work between them accordingly and exchange the data as necessary.
- Think about and implement additional optimisations to the code (memoisation, memory access patterns, asynchronous data transfers, overlapping computation, code fusion, etc.)



