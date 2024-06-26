#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>
#include "cuco_hashmap.h"

namespace py = pybind11;
using namespace pycuco;

PYBIND11_MODULE(PyCUCOLib, m) {
  py::class_<CUCOHashmapWrapper>(m, "CUCOStaticHashmap")
      .def(py::init<torch::Tensor, torch::Tensor, double>())
      .def("query", &CUCOHashmapWrapper::query)
      .def("capacity", &CUCOHashmapWrapper::get_capacity)
      .def("memory_usage", &CUCOHashmapWrapper::get_memory_usage);
}