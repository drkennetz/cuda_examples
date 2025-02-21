#include <thrust/sort.h>
#include <thrust/device_vector.h>

struct plus_one_functor {
    __device__
    int operator()(int x) {
        return x + 1;
    }
};

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

    // Transform
    {
        thrust::device_vector<int> d_vec(10);
        thrust::sequence(
            thrust::device,
            d_vec.begin(),
            d_vec.end()
        );
        print_vector(d_vec);
        // Now, we will add 1 to every element
        thrust::transform(
            thrust::device,
            d_vec.begin(),
            d_vec.end(),
            d_vec.begin(),
            plus_one_functor()
        );
        print_vector(d_vec);
    }

    // Merge
    {
        std::vector<int> h_vec1 = {1, 2, 3, 4, 5};
        std::vector<int> h_vec2 = {6, 7, 8, 9, 10};
        thrust::device_vector<int> d_vec1 = h_vec1;
        thrust::device_vector<int> d_vec2 = h_vec2;
        thrust::device_vector<int> merged_vec(10);
        thrust::merge(
            thrust::device,
            d_vec1.begin(),
            d_vec1.end(),
            d_vec2.begin(),
            d_vec2.end(),
            merged_vec.begin()
        );
        print_vector(merged_vec);
    }
}
