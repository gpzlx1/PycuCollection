#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>
#include "cuco_hashmap.h"

namespace py = pybind11;

PYBIND11_MODULE(PyCUCOLib, m) {
  py::class_<CUCOHashmapWrapper>(m, "CUCOStaticHashmap")
      .def(py::init<torch::Tensor, torch::Tensor, double>())
      .def("query", &CUCOHashmapWrapper::query);
}