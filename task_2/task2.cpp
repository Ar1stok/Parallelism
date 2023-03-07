#include <iostream>
#include <cstring>
#include <sstream>
#include <math.h>
#include <cmath>

#ifdef OPENACC__
#include <openacc.h>
#endif
#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif

#define at(arr, x, y) (arr[(x)*size+(y)]) 

void initArrays(double* mainArr, double* subArr, int& size){
    std::memset(mainArr, 0, sizeof(double) * size * size);
    for(int i = 0; i < size; i++)
    {
        at(mainArr, 0, i) = 10 / size * i + 10;
        at(mainArr, i, 0) = 10 / size * i + 10;
        at(mainArr, size-1, i) = 10 / size * i + 20;
        at(mainArr, i, size-1) = 10 / size * i + 20;

        at(subArr, 0, i) = 10 / size * i + 10;
        at(subArr, i, 0) = 10 / size * i + 10;
        at(subArr, size-1, i) = 10 / size * i + 20;
        at(subArr, i, size-1) = 10 / size * i + 20;
    }
    std::memcpy(subArr, mainArr, sizeof(double) * size * size);
}

constexpr int ITERS_UP = 5;

int main(int argc, char *argv[]){

    double eps = 1E-6;
    int iterations = 1E6;
    int size = 10;

    for(int arg = 0; arg < argc; arg++){
        std::stringstream stream;
        if(strcmp(argv[arg], "-eps") == 0){
            stream << argv[arg+1];
            stream >> eps;
        }
        else if(strcmp(argv[arg], "-i") == 0){
            stream << argv[arg+1];
            stream >> iterations;
        }
        else if(strcmp(argv[arg], "-s") == 0){
            stream << argv[arg+1];
            stream >> size;
        }
    }

    std::cout << "Current settings: " << "eps - " << eps << ", max iteration - " << 
    iterations << ", size - " << size << 'x' << size << std::endl;
 
    double* F = new double[size*size];
    double* Fnew = new double[size*size];

    initArrays(F, Fnew, size);

    double error = 0;
    int iteration = 0;
    int iters_up = 0;

    #pragma acc enter data copyin(Fnew[:size*size], F[:size*size], error)

#ifdef NVPROF_
    nvtxRangePush("MainCycle");
#endif
    do {
        #pragma acc parallel present(error) async
        {
            error = 0;
        }

        #pragma acc parallel loop collapse(2) present(Fnew[:size*size], F[:size*size], error) reduction(max:error) vector_length(128) async(0)
        for(int x = 1; x < size-1; x++)
        {
            for(int y = 1; y < size-1; y++)
            {
                at(Fnew, x, y) = 0.25 * (at(F, x+1, y) + at(F, x-1, y) + at(F, x, y-1) + at(F, x, y+1));
                error = fmax(error, fabs(at(Fnew, x, y) - at(F, x, y)));
            }
        }

        double* swap = F;
        F = Fnew;
        Fnew = swap;

#ifdef OPENACC__
        acc_attach((void**)F);
        acc_attach((void**)Fnew);
#endif
        if(iters_up >= ITERS_UP && iteration < iterations)
        {
            #pragma acc update self(error) wait
            iters_up = -1;
        }
        else
        {
            error = 1;
        }
        iteration++;
        iters_up++;

    } while(iteration < iterations && error > eps);


#ifdef NVPROF_
    nvtxRangePop();
#endif

    #pragma acc exit data delete(Fnew[:size*size]) copyout(F[:size*size], error)

    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Error: " << error << std::endl;

    delete[] F;
    delete[] Fnew;

    return 0;
}