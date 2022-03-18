// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "opencl_utils.h"
#include "opencl_forward_decl.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/graph/constants.h"

/* Notes about OpenCL object lifetime

OpenCL have reference counted (RC) object management, however, it is impossible
to query the current counter. As a result, it makes it difficualt to *share*
created object, especially of programs and kernels, which are slow to create
and resulting loooong startup time on the first run of a session. So we are not
relying on the internal RC mechanism.

- cl_device_id, cl_context and cl_command_queue

  lifetime are bounded to EP

- cl_mem

  is created and managed by OpenCLBufferAllocator and OpenCLImage2DAllocator

- cl_program and cl_kernel

  is created and managed by OpenCLKernelManager

*/

namespace onnxruntime {

// Information needed to construct OpenCL execution providers.
struct OpenCLExecutionProviderInfo {
  bool use_fp16;
};

using IAllocatorUniquePtrToClMem = IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>>;

// Logical device representation.
class OpenCLExecutionProvider : public IExecutionProvider {
 public:
  explicit OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpenCLExecutionProvider);
  virtual ~OpenCLExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) override;

  cl_device_id GetOpenCLDevice() const { return dev_; }
  cl_context GetOpenCLContext() const { return ctx_; }
  cl_command_queue GetCommandQueue() const { return cmd_queue_; }

  IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>> GetScratchBuffer(size_t nbytes) const;
  IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>> GetScratchImage2D(const opencl::Image2DDesc& desc) const;

  bool UseFp16() const { return use_fp16_; }

  Status AfterCLLaunch() const;

  const opencl::OpenCLProgramManager* GetProgramManager() const;
  opencl::OpenCLProgramManager* GetProgramManager();

 private:
  Status InitOpenCLContext();
  void DisableFp16() { use_fp16_ = false; }
  static bool ShouldFlushAfterLaunch(const std::string& device_name);

  cl_device_id dev_;
  cl_context ctx_;
  cl_command_queue cmd_queue_;
  bool use_fp16_;
  bool flush_after_launch_;

 private:
  std::unique_ptr<opencl::OpenCLProgramManager> program_manager_;

  // IDataTransfer is a lightweight interface with std::unique_ptr as its
  // return value. Bind kernels to it directly will cause the kernel being
  // created from time to time. So we move the kernels here.
  std::unique_ptr<opencl::OpenCLKernelHolder> copy_kernels_;
  void InitCopyKernels();

#ifdef TRACY_ENABLE
  TracyCLCtx tracy_cl_ctx_;

 public:
  TracyCLCtx GetTracyCLContext() { return tracy_cl_ctx_; }
  const std::remove_pointer_t<TracyCLCtx>* GetTracyCLContext() const { return tracy_cl_ctx_; }
#endif
};

}  // namespace onnxruntime
