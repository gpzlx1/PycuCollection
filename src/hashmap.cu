#include <iostream>
#include <bght/bcht.hpp>
#include "hashmap.h"


torch::Tensor add(torch::Tensor a, torch::Tensor b){
  return a + b;
}

using pair_type = bght::pair<int32_t, int32_t>;

__global__ void InitPair(pair_type* pair, int32_t* key, int32_t* value,
                         int64_t capacity) {
  for (int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
       thread_idx < capacity; thread_idx += gridDim.x * blockDim.x) {
    pair[thread_idx].first = key[thread_idx];
    pair[thread_idx].second = value[thread_idx];
  }
}

BGHTHashmap::BGHTHashmap(torch::Tensor keys, torch::Tensor values) {
  int64_t capacity = keys.numel() * 1.2;

  pair_type* pair_;
  cudaMalloc(&pair_, int64_t(int64_t(keys.numel()) * sizeof(pair_type)));
  int block_size = 1024;
  int grid_size = (keys.numel() + block_size - 1) / block_size;
  InitPair<<<grid_size, block_size>>>(pair_, keys.data_ptr<int32_t>(),
                                      values.data_ptr<int32_t>(), keys.numel());

  // create hashmap
  auto hashmap_ptr = new bght::bcht<int32_t, int32_t>(capacity, -1, -1);
  hashmap_ptr->insert(pair_, pair_ + keys.numel());

  cudaFree(pair_);

  hashmap_ = hashmap_ptr;
}

BGHTHashmap::~BGHTHashmap() {
  delete static_cast<bght::bcht<int32_t, int32_t>*>(hashmap_);

}

torch::Tensor BGHTHashmap::query(torch::Tensor requests) {
  CHECK(requests.device().is_cuda());

  int64_t numel = requests.numel();

  torch::Tensor result = torch::full_like(requests, -1, torch::kInt32);

  bght::bcht<int32_t, int32_t>* hashmap_ptr =
      static_cast<bght::bcht<int32_t, int32_t>*>(hashmap_);

  
  hashmap_ptr->find(requests.data_ptr<int32_t>(),
                    requests.data_ptr<int32_t>() + numel,
                    result.data_ptr<int32_t>());

  return result;
}


