// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "opencl_utils.h"
#include "opencl_forward_decl.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct OpenCL execution providers.
struct OpenCLExecutionProviderInfo {
  bool use_fp16;
};

// Logical device representation.
class OpenCLExecutionProvider : public IExecutionProvider {
  friend class opencl::OpenCLDataTransfer;

 public:
  explicit OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info);
  OpenCLExecutionProvider(OpenCLExecutionProvider&&) noexcept;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpenCLExecutionProvider);
  virtual ~OpenCLExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) override;

  cl::Device GetOpenCLDevice() const { return dev_; }
  cl::Context GetOpenCLContext() const { return ctx_; }
  cl::CommandQueue GetCommandQueue() const { return cmd_queue_; }

  IAllocatorUniquePtr<cl::Buffer> GetScratchBuffer(size_t nbytes) const;
  IAllocatorUniquePtr<cl::Image2D> GetScratchImage2D(opencl::Image2DDesc desc) const;

  bool UseFp16() const { return use_fp16_; }

 private:
  Status InitOpenCLContext();
  void DisableFp16() { use_fp16_ = false; }

  cl::Device dev_;
  cl::Context ctx_;
  cl::CommandQueue cmd_queue_;
  bool use_fp16_;

 private:
  // IDataTransfer is a lightweight interface with std::unique_ptr as its
  // return value. Bind kernels to it directly will cause the kernel being
  // created from time to time. So we move the kernels here.
  std::unique_ptr<opencl::OpenCLKernelHolder> copy_kernels_;
  void InitCopyKernels();
};

}  // namespace onnxruntime
