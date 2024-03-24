#pragma once
#include <torch/extension.h>

class Hashmap {};

class CUCOHashmapWrapper {
 public:
  CUCOHashmapWrapper(torch::Tensor keys, torch::Tensor values,
                     double load_factor);
  ~CUCOHashmapWrapper() { delete map_; };
  torch::Tensor query(torch::Tensor requests);
  // int64_t get_memory_usage();

 private:
  Hashmap* map_;
};