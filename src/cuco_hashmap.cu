#include <cuco/static_map.cuh>
#include <iostream>
#include "cuco_hashmap.h"

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

template <typename Key, typename Value>
class CUCOHashmap : public Hashmap {
 public:
  using map_type = cuco::static_map<
      Key, Value, std::size_t, cuda::thread_scope_device, thrust::equal_to<Key>,
      cuco::linear_probing<4, cuco::default_hash_function<Key>>,
      cuco::cuda_allocator<cuco::pair<Key, Value>>, cuco::storage<1>>;

  CUCOHashmap(torch::Tensor keys, torch::Tensor values, double load_factor) {
    Key constexpr empty_key_sentinel = -1;
    Value constexpr empty_value_sentinel = -1;

    int64_t numel = keys.numel();
    std::size_t const capacity = std::ceil(numel / load_factor);

    // Create a cuco::static_map
    map_ = new map_type(capacity, cuco::empty_key{empty_key_sentinel},
                        cuco::empty_value{empty_value_sentinel});
    auto zipped = thrust::make_zip_iterator(
        thrust::make_tuple(keys.data_ptr<Key>(), values.data_ptr<Value>()));
    map_->insert(zipped, zipped + numel);
  };

  ~CUCOHashmap() { delete map_; };

  torch::Tensor query(torch::Tensor requests) {
    int64_t numel = requests.numel();
    torch::Tensor result = torch::full_like(requests, -1, value_options_);
    map_->find(requests.data_ptr<Key>(), requests.data_ptr<Key>() + numel,
               result.data_ptr<Value>());
    return result;
  };

  int64_t get_memory_usage() { return memory_usage_; }

 private:
  torch::TensorOptions key_options_;
  torch::TensorOptions value_options_;
  int64_t memory_usage_;
  int64_t capacity_;
  map_type* map_;
};

/*
template <typename Key, typename Value>
CUCOHashmap<Key, Value>::CUCOHashmap(torch::Tensor keys, torch::Tensor values,
                                     double load_factor) {

}

template <typename Key, typename Value>
torch::Tensor CUCOHashmap<Key, Value>::query(torch::Tensor requests) {

}
*/

CUCOHashmapWrapper::CUCOHashmapWrapper(torch::Tensor keys, torch::Tensor values,
                                       double load_factor) {
  map_ = new CUCOHashmap<int32_t, int32_t>(keys, values, load_factor);
}

torch::Tensor CUCOHashmapWrapper::query(torch::Tensor requests) {
  CUCOHashmap<int32_t, int32_t>* map = (CUCOHashmap<int32_t, int32_t>*)map_;
  return map->query(requests);
}
