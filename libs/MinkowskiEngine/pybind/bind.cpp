#include "coordinate_map_key.hpp"
#include "types.hpp"
#include "utils.hpp"

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace minkowski {
namespace py = pybind11;
}

namespace py = pybind11;

void initialize_non_templated_classes(py::module &m) {
  // Enums
  py::enum_<minkowski::GPUMemoryAllocatorBackend::Type>(
      m, "GPUMemoryAllocatorType")
      .value("PYTORCH", minkowski::GPUMemoryAllocatorBackend::Type::PYTORCH)
      .value("CUDA", minkowski::GPUMemoryAllocatorBackend::Type::CUDA)
      .export_values();

  py::enum_<minkowski::CUDAKernelMapMode::Mode>(m, "CUDAKernelMapMode")
      .value("MEMORY_EFFICIENT",
             minkowski::CUDAKernelMapMode::Mode::MEMORY_EFFICIENT)
      .value("SPEED_OPTIMIZED",
             minkowski::CUDAKernelMapMode::Mode::SPEED_OPTIMIZED)
      .export_values();

  py::enum_<minkowski::MinkowskiAlgorithm::Mode>(m, "MinkowskiAlgorithm")
      .value("DEFAULT", minkowski::MinkowskiAlgorithm::Mode::DEFAULT)
      .value("MEMORY_EFFICIENT",
             minkowski::MinkowskiAlgorithm::Mode::MEMORY_EFFICIENT)
      .value("SPEED_OPTIMIZED",
             minkowski::MinkowskiAlgorithm::Mode::SPEED_OPTIMIZED)
      .export_values();

  py::enum_<minkowski::CoordinateMapBackend::Type>(m, "CoordinateMapType")
      .value("CPU", minkowski::CoordinateMapBackend::Type::CPU)
      .value("CUDA", minkowski::CoordinateMapBackend::Type::CUDA)
      .export_values();

  py::enum_<minkowski::RegionType::Type>(m, "RegionType")
      .value("HYPER_CUBE", minkowski::RegionType::Type::HYPER_CUBE)
      .value("HYPER_CROSS", minkowski::RegionType::Type::HYPER_CROSS)
      .value("CUSTOM", minkowski::RegionType::Type::CUSTOM)
      .export_values();

  py::enum_<minkowski::PoolingMode::Type>(m, "PoolingMode")
      .value("LOCAL_SUM_POOLING",
             minkowski::PoolingMode::Type::LOCAL_SUM_POOLING)
      .value("LOCAL_AVG_POOLING",
             minkowski::PoolingMode::Type::LOCAL_AVG_POOLING)
      .value("LOCAL_MAX_POOLING",
             minkowski::PoolingMode::Type::LOCAL_MAX_POOLING)
      .value("GLOBAL_SUM_POOLING_DEFAULT",
             minkowski::PoolingMode::Type::GLOBAL_SUM_POOLING_DEFAULT)
      .value("GLOBAL_AVG_POOLING_DEFAULT",
             minkowski::PoolingMode::Type::GLOBAL_AVG_POOLING_DEFAULT)
      .value("GLOBAL_MAX_POOLING_DEFAULT",
             minkowski::PoolingMode::Type::GLOBAL_MAX_POOLING_DEFAULT)
      .value("GLOBAL_SUM_POOLING_KERNEL",
             minkowski::PoolingMode::Type::GLOBAL_SUM_POOLING_KERNEL)
      .value("GLOBAL_AVG_POOLING_KERNEL",
             minkowski::PoolingMode::Type::GLOBAL_AVG_POOLING_KERNEL)
      .value("GLOBAL_MAX_POOLING_KERNEL",
             minkowski::PoolingMode::Type::GLOBAL_MAX_POOLING_KERNEL)
      .value("GLOBAL_SUM_POOLING_PYTORCH_INDEX",
             minkowski::PoolingMode::Type::GLOBAL_SUM_POOLING_PYTORCH_INDEX)
      .value("GLOBAL_AVG_POOLING_PYTORCH_INDEX",
             minkowski::PoolingMode::Type::GLOBAL_AVG_POOLING_PYTORCH_INDEX)
      .value("GLOBAL_MAX_POOLING_PYTORCH_INDEX",
             minkowski::PoolingMode::Type::GLOBAL_MAX_POOLING_PYTORCH_INDEX)
      .export_values();

  py::enum_<minkowski::BroadcastMode::Type>(m, "BroadcastMode")
      .value("ELEMENTWISE_ADDITON",
             minkowski::BroadcastMode::Type::ELEMENTWISE_ADDITON)
      .value("ELEMENTWISE_MULTIPLICATION",
             minkowski::BroadcastMode::Type::ELEMENTWISE_MULTIPLICATION)
      .export_values();

  py::enum_<minkowski::ConvolutionMode::Type>(m, "ConvolutionMode")
      .value("DEFAULT", minkowski::ConvolutionMode::Type::DEFAULT)
      .value("DIRECT_GEMM", minkowski::ConvolutionMode::Type::DIRECT_GEMM)
      .value("COPY_GEMM", minkowski::ConvolutionMode::Type::COPY_GEMM)
      .export_values();

  // Classes
  py::class_<minkowski::CoordinateMapKey>(m, "CoordinateMapKey")
      .def(py::init<minkowski::default_types::size_type>())
      .def(py::init<minkowski::default_types::stride_type, std::string>())
      .def("__repr__", &minkowski::CoordinateMapKey::to_string)
      .def("__hash__", &minkowski::CoordinateMapKey::hash)
      .def("is_key_set", &minkowski::CoordinateMapKey::is_key_set)
      .def("get_coordinate_size",
           &minkowski::CoordinateMapKey::get_coordinate_size)
      .def("get_key", &minkowski::CoordinateMapKey::get_key)
      .def("set_key", (void (minkowski::CoordinateMapKey::*)(
                          minkowski::default_types::stride_type, std::string)) &
                          minkowski::CoordinateMapKey::set_key)
      .def("set_key", (void (minkowski::CoordinateMapKey::*)(
                          minkowski::coordinate_map_key_type const &)) &
                          minkowski::CoordinateMapKey::set_key)
      .def("get_tensor_stride", &minkowski::CoordinateMapKey::get_tensor_stride)
      .def("__eq__", [](const minkowski::CoordinateMapKey &self, const minkowski::CoordinateMapKey &other)
                     {
                       return self == other;
                     });
}
