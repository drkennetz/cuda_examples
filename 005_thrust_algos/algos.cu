#include <thrust/sort.h>
#include <thrust/device_vector.h>

void print_vector(thrust::device_vector<int> v) {
    thrust::for_each(v.begin(), v.end(), [] __host__ __device__(int val) {
        printf("%d ", val);
    });
    std::cout << std::endl;
}

int main() {
    // Sorting
    {
        std::vector<int> h_vec{3, 1, 4, 2, 8, 5, 9, 12, 6};
        thrust::device_vector<int> d_vec = h_vec;
        thrust::sort(
            thrust::device,
            d_vec.begin(),
            d_vec.end()
        );
        print_vector(d_vec);
    }

    // Filling
    {
        thrust::device_vector<int> d_vec(10);
        thrust::fill(
            thrust::device,
            d_vec.begin(),
            d_vec.end(),
            101
        );
        print_vector(d_vec);
    }

    // Sequence
    {
        thrust::device_vector<int> d_vec(10);
        thrust::sequence(
            thrust::device,
            d_vec.begin(),
            d_vec.end()
        );  
        print_vector(d_vec);
    }
}
