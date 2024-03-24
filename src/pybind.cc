#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>
#include "bght_hashmap.h"

namespace py = pybind11;

PYBIND11_MODULE(BGHTLib, m) {
  py::class_<BGHTHashmap>(m, "BGHTHashmap")
      .def(py::init<torch::Tensor, torch::Tensor>())
      .def("memory_usage", &BGHTHashmap::get_memory_usage)
      .def("query", &BGHTHashmap::query);
}