#ifndef BIFEAT_HASHMAP_H
#define BIFEAT_HASHMAP_H

#include <pybind11/pybind11.h>
#include <torch/extension.h>

class BGHTHashmap {
 public:
  BGHTHashmap(torch::Tensor keys, torch::Tensor values);
  ~BGHTHashmap();

  torch::Tensor query(torch::Tensor requests);
  int64_t get_memory_usage() { return memory_usage_; }

 private:
  int64_t memory_usage_;
  int64_t capacity_;
  void *hashmap_;
};

#endif