# Gale-Shapley Stable Matching - Parallel Implementations

This project implements the Gale-Shapley stable matching algorithm across four parallel architectures:

- **Serial Version** (Baseline)
- **Shared Address Space (OpenMP)**
- **Message Passing Interface (MPI)**
- **CUDA (GPU Parallelism)**

---

## Compilation and Execution

### Serial Version
```bash
make clean
make
./gale_shapley_serial -n <number of participants per group> -s <seed>
```

### Shared Address Space Version
```bash
g++ -std=c++17 -fopenmp -O3 -o galeshapley_shared galeshapley_shared.cpp
./galeshapley_shared -n <num of participants of one group> -s <seed> -t <num of threads>
```

### Message Passing Version
```bash
mpic++ -std=c++17 -O3 -Wall -o galeshapley_mpi galeshapley_mpi.cpp
mpirun -np <nproc> ./galeshapley_mpi -n <num of participants of one group> -s <seed>
```

### CUDA Version
```bash
nvcc -std=c++17 -O2 -o galeshapley_cuda galeshapley_cuda.cu
./galeshapley_cuda -n <num of participants of one group>
```

