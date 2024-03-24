#include <thrust/iterator/zip_iterator.h>
#include <cuco/static_map.cuh>
#include <iostream>

#include "common.h"
#include "cuco_hashmap.h"

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

    // Set property
    key_options_ = keys.options();
    value_options_ = values.options();
    capacity_ = capacity;
    memory_usage_ = 100;  // for test
  };

  ~CUCOHashmap() { delete map_; };

  torch::Tensor query(torch::Tensor requests) {
    int64_t numel = requests.numel();
    torch::Tensor result = torch::full_like(requests, -1, value_options_);
    map_->find(requests.data_ptr<Key>(), requests.data_ptr<Key>() + numel,
               result.data_ptr<Value>());
    return result;
  };

 private:
  torch::TensorOptions key_options_;
  torch::TensorOptions value_options_;
  // int64_t memory_usage_;
  // int64_t capacity_;
  map_type* map_;
};

CUCOHashmapWrapper::CUCOHashmapWrapper(torch::Tensor keys, torch::Tensor values,
                                       double load_factor) {
  CHECK_CUDA(keys);
  CHECK_CUDA(values);
  key_type_ = keys.dtype();
  value_type_ = values.dtype();

  INTEGER_TYPE_SWITCH(key_type_, Key, {
    INTEGER_TYPE_SWITCH(value_type_, Value, {
      map_ = new CUCOHashmap<Key, Value>(keys, values, load_factor);
    });
  });
}

torch::Tensor CUCOHashmapWrapper::query(torch::Tensor requests) {
  CHECK_CUDA(requests);
  INTEGER_TYPE_SWITCH(key_type_, Key, {
    INTEGER_TYPE_SWITCH(value_type_, Value, {
      auto map = (CUCOHashmap<Key, Value>*)map_;
      return map->query(requests.to(key_type_));
    });
  });

  return torch::Tensor();
}
