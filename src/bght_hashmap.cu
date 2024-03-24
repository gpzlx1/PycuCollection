#include <bght/bcht.hpp>
#include <iostream>
#include "bght_hashmap.h"

using pair_type = bght::pair<int32_t, int32_t>;

__global__ void InitPair(pair_type *pair, int32_t *key, int32_t *value,
                         int64_t capacity) {
  for (int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
       thread_idx < capacity; thread_idx += gridDim.x * blockDim.x) {
    pair[thread_idx].first = key[thread_idx];
    pair[thread_idx].second = value[thread_idx];
  }
}

size_t size = 0;

template <class T>
struct my_allocator {
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;

  template <class U>
  struct rebind {
    typedef my_allocator<U> other;
  };
  my_allocator() = default;
  template <class U>
  constexpr my_allocator(const my_allocator<U> &) noexcept {}
  T *allocate(std::size_t n) {
    void *p = nullptr;
    cuda_try(cudaMalloc(&p, n * sizeof(T)));
    std::cout << "Memory usage: " << n * sizeof(T) << std::endl;
    size += n * sizeof(T);
    return static_cast<T *>(p);
  }
  void deallocate(T *p, std::size_t n) noexcept { cuda_try(cudaFree(p)); }
};

BGHTHashmap::BGHTHashmap(torch::Tensor keys, torch::Tensor values) {
  int64_t capacity = keys.numel() * 1.2;

  pair_type *pair_;
  cudaMalloc(&pair_, int64_t(int64_t(keys.numel()) * sizeof(pair_type)));
  int block_size = 1024;
  int grid_size = (keys.numel() + block_size - 1) / block_size;
  InitPair<<<grid_size, block_size>>>(pair_, keys.data_ptr<int32_t>(),
                                      values.data_ptr<int32_t>(), keys.numel());

  // create hashmap
  size = 0;
  auto allocator = my_allocator<char>();
  auto hashmap_ptr =
      new bght::bcht<int32_t, int32_t, bght::MurmurHash3_32<int32_t>,
                     bght::equal_to<int32_t>, cuda::thread_scope_device,
                     my_allocator<char>, 16>(capacity, -1, -1, allocator);
  hashmap_ptr->insert(pair_, pair_ + keys.numel());
  memory_usage_ = size;

  // free and return
  cudaFree(pair_);
  hashmap_ = hashmap_ptr;
}

BGHTHashmap::~BGHTHashmap() {
  delete static_cast<bght::bcht<int32_t, int32_t> *>(hashmap_);
}

torch::Tensor BGHTHashmap::query(torch::Tensor requests) {
  CHECK(requests.device().is_cuda());

  int64_t numel = requests.numel();

  torch::Tensor result = torch::full_like(requests, -1, torch::kInt32);

  bght::bcht<int32_t, int32_t> *hashmap_ptr =
      static_cast<bght::bcht<int32_t, int32_t> *>(hashmap_);

  hashmap_ptr->find(requests.data_ptr<int32_t>(),
                    requests.data_ptr<int32_t>() + numel,
                    result.data_ptr<int32_t>());

  return result;
}
