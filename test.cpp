#include <iostream>
#include <cmath>

#define arr_elem 10000000

#ifdef d_double
using nspace = double;
#else
using nspace = float;
#endif // d_double


const nspace pi = std::acos(-1);

int main(int argc, char const* argv[])
{
    nspace sum = 0;
    nspace* array;
    array = new nspace[arr_elem];

#pragma acc enter data create(array[0:arr_elem],sum)

#pragma acc parallel loop present(array[0:arr_elem])
    for (int i = 0; i < arr_elem; ++i)
    {
        array[i] = sin(2*pi*i/arr_elem);
    }

#pragma acc parallel loop present(array[0:arr_elem],sum) reduction(+:sum)
    for (int i = 0; i < arr_elem; ++i)
    {
        sum += array[i];
    }

#pragma acc exit data delete(array[0:arr_elem]) copyout(sum)

    std::cout << sum << std::endl;

    return 0;
}