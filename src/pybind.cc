#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>
#include "hashmap.h"

namespace py = pybind11;

PYBIND11_MODULE(BGHTLib, m) {
  py::class_<BGHTHashmap>(m, "BGHTHashmap")
      .def(py::init<torch::Tensor, torch::Tensor>())
      .def("query", &BGHTHashmap::query);

  m.def("add", &add, py::arg("a"), py::arg("b"));
  
}