// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

#include <unordered_map>
#include <list>
#include <functional>
#include <variant>
#include "opencl_utils.h"

namespace onnxruntime {
namespace opencl {

constexpr auto BufferAllocatorName = "OpenCL_Buffer";
constexpr auto Image2DAllocatorName = "OpenCL_Image2D";
constexpr auto CPUAllocatorName = "OpenCL_CPU";
constexpr auto CPUInputAllocatorName = "OpenCL_CPU_Input";

class OpenCLBufferAllocator : public IAllocator {
 public:
  struct Metadata {
    size_t size;
    MemoryKind kind;
  };

  explicit OpenCLBufferAllocator(cl_context ctx);
  ~OpenCLBufferAllocator() override;

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  cl_context ctx_;
  // FIXME: better caching, cache for kernel benchmark at the moment
  std::unordered_map<void*, Metadata> meta_;
  std::unordered_map<size_t, std::list<void*>> cache_;
};

}  // namespace opencl
}  // namespace onnxruntime

template <>
struct std::hash<onnxruntime::TensorShape> {
  size_t operator()(const onnxruntime::TensorShape& shape) const {
    auto dims = shape.GetDims();
    using DT = decltype(dims);
    auto h = std::hash<DT::value_type>{}(0);
    for (const auto& s : dims) {
      h ^= std::hash<DT::value_type>{}(s);
    }
    return h;
  }
};

namespace onnxruntime {
namespace opencl {

class OpenCLImage2DAllocator : public IAllocator {
 public:
  struct Metadata {
    TensorShape shape;
    MemoryKind kind;
  };

  explicit OpenCLImage2DAllocator(cl_context ctx, bool use_fp16);
  ~OpenCLImage2DAllocator() override;

  void* Alloc(size_t size) override;
  void* Alloc(const TensorShape& shape) override;
  void Free(void* p) override;

 private:
  cl_context ctx_;
  bool use_fp16_;

  // FIXME: better caching, cache for kernel benchmark at the moment
  std::unordered_map<void*, Metadata> meta_;
  std::unordered_map<TensorShape, std::list<void*>> cache_;
};

}  // namespace opencl
}  // namespace onnxruntime
