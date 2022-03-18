// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_execution_provider.h"
#include "opencl_allocator.h"
#include "opencl_program_manager.h"
#include "opencl_data_transfer.h"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"

#include <array>
#include <utility>

// Add includes of kernel implementations
#include "memcpy_kernel.h"
#include "core/providers/opencl/math/clip.h"
#include "core/providers/opencl/math/elementwise.h"
#include "core/providers/opencl/nn/conv.h"
#include "core/providers/opencl/nn/global_average_pool.h"
#include "core/providers/opencl/nn/relu.h"
#include "core/providers/opencl/nn/max_pool.h"
#include "core/providers/opencl/nn/concat.h"
#include "core/providers/opencl/tensor/shape.h"
#include "core/providers/opencl/tensor/resize.h"

namespace onnxruntime {
namespace opencl {

Status RegisterOpenCLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 15, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, AddRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 6, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 12, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 6, 12, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 8, 11, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 12, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, 12, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kMSDomain, 1, FusedConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, 12, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Resize)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    VLOGS_DEFAULT(1) << "[CL] RegisterOpenCLKernels...";
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      VLOGS_DEFAULT(0) << "[CL]  register kernel name: " << info.kernel_def->OpName() << ", domain: " << info.kernel_def->Domain();
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetOpenCLKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterOpenCLKernels(*kernel_registry));
  return kernel_registry;
}

}  // namespace opencl

OpenCLExecutionProvider::OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info)
    : IExecutionProvider(kOpenCLExecutionProvider), use_fp16_(info.use_fp16) {
  Status status;
#ifdef CL3W_ENABLE
  if (cl3wInit() != CL3W_OK) {
    ORT_THROW("cl3w initialization failure.");
  }
#endif
  ORT_THROW_IF_ERROR(InitOpenCLContext());
  program_manager_ = std::make_unique<opencl::OpenCLProgramManager>(this);
  InitCopyKernels();

#ifdef TRACY_ENABLE
  tracy_cl_ctx_ = TracyCLContext(ctx_, dev_);
#endif
}

OpenCLExecutionProvider::OpenCLExecutionProvider(OpenCLExecutionProvider&& provider) noexcept
    : IExecutionProvider(kOpenCLExecutionProvider), use_fp16_(provider.use_fp16_) {
  std::swap(dev_, provider.dev_);
  std::swap(ctx_, provider.ctx_);
  std::swap(cmd_queue_, provider.cmd_queue_);

  std::swap(use_fp16_, provider.use_fp16_);
  std::swap(flush_after_launch_, provider.flush_after_launch_);

  std::swap(program_manager_, provider.program_manager_);
  std::swap(copy_kernels_, provider.copy_kernels_);

#ifdef TRACY_ENABLE
  std::swap(tracy_cl_ctx_, provider.tracy_cl_ctx_);
#endif
}

OpenCLExecutionProvider::~OpenCLExecutionProvider() {
  // FIXME: kernel manager should release all managed kernels and programs

#ifdef TRACY_ENABLE
  TracyCLCollect(tracy_cl_ctx_);
  TracyCLDestroy(tracy_cl_ctx_);
#endif

  clReleaseCommandQueue(cmd_queue_);
  clReleaseDevice(dev_);
  clReleaseContext(ctx_);
}

std::shared_ptr<KernelRegistry> OpenCLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = opencl::GetOpenCLKernelRegistry();
  return kernel_registry;
}

Status OpenCLExecutionProvider::InitOpenCLContext() {
  cl_uint num_platforms;
  ORT_RETURN_IF_CL_ERROR(clGetPlatformIDs(0, nullptr, &num_platforms));
  // NOTE: the EP is in construction, the logger_ is not registered
  LOGS_DEFAULT(VERBOSE) << "[CL] num platforms: " << num_platforms;
  ORT_RETURN_IF_NOT(num_platforms > 0, "Cannot find OpenCL platform.");

  std::vector<cl_platform_id> platforms(num_platforms);
  ORT_RETURN_IF_CL_ERROR(clGetPlatformIDs(platforms.size(), platforms.data(), nullptr));
  int selected_platform_idx = -1;
  // FIXME: add platform selection logic
  for (int i = 0; i < platforms.size(); i++) {
    size_t ret_size;
    ORT_RETURN_IF_CL_ERROR(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, nullptr, &ret_size));
    std::string vendor(ret_size, '\0');
    ORT_RETURN_IF_CL_ERROR(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, vendor.size(), vendor.data(), nullptr));
    LOGS_DEFAULT(VERBOSE) << "[CL] platform vendor: " << vendor;
    if (vendor == "Oclgrind") {
      LOGS_DEFAULT(INFO) << "[CL] platform " << vendor << " selected";
      selected_platform_idx = 1;
      break;
    }
  }
  if (selected_platform_idx == -1) {
    LOGS_DEFAULT(INFO) << "[CL] default platform selected";
    selected_platform_idx = 0;
  }
  auto* selected_platform = platforms[selected_platform_idx];

  cl_int err{};
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)selected_platform, 0};
  ctx_ = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, /*pfn_notify=*/nullptr, /*user_data=*/nullptr, &err);
  ORT_RETURN_IF_CL_ERROR(err);

  size_t ret_size;
  ORT_RETURN_IF_CL_ERROR(clGetContextInfo(ctx_, CL_CONTEXT_DEVICES, 0, nullptr, &ret_size));
  std::vector<cl_device_id> devices(ret_size);
  ORT_RETURN_IF_CL_ERROR(clGetContextInfo(ctx_, CL_CONTEXT_DEVICES, devices.size(), devices.data(), nullptr));
  LOGS_DEFAULT(VERBOSE) << "[CL] num devices: " << devices.size();
  ORT_RETURN_IF(devices.empty(), "Cannot find OpenCL device.");
  dev_ = devices[0];

  auto GetDeviceInfo = [=](cl_device_info info_name) -> std::string {
    size_t ret_size;
    ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(dev_, info_name, 0, nullptr, &ret_size));
    std::string ret(ret_size, '\0');
    ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(dev_, info_name, ret.size(), ret.data(), nullptr));
    return ret;
  };

  // NOTE: use stdout for mobile
  // FIXME: use logger latter
  auto device_name = GetDeviceInfo(CL_DEVICE_NAME);
  LOGS_DEFAULT(INFO) << "[CL] device name: " << device_name << "\n";
  LOGS_DEFAULT(VERBOSE) << "[CL] device vendor: " << GetDeviceInfo(CL_DEVICE_VENDOR) << "\n";
  LOGS_DEFAULT(VERBOSE) << "[CL] device version: " << GetDeviceInfo(CL_DEVICE_VERSION) << "\n";
  auto exts = GetDeviceInfo(CL_DEVICE_EXTENSIONS);
  LOGS_DEFAULT(VERBOSE) << "[CL] device extensions: " << exts << std::endl;
  bool has_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
  if (!has_fp16 && UseFp16()) {
    LOGS_DEFAULT(WARNING) << "[CL] FP16 is requested, but is not supported by the device!";
    DisableFp16();
  }
  flush_after_launch_ = ShouldFlushAfterLaunch(device_name);
  LOGS_DEFAULT(INFO) << "[CL] FP16: " << UseFp16() << "\n";
  LOGS_DEFAULT(INFO) << "[CL] clFlush after launch: " << flush_after_launch_ << "\n";

#ifdef TRACY_ENABLE
  cmd_queue_ = clCreateCommandQueue(ctx_, dev_, CL_QUEUE_PROFILING_ENABLE, &err);
#else
  cmd_queue_ = clCreateCommandQueue(ctx_, dev_, /*properties=*/0, &err);
#endif
  ORT_RETURN_IF_CL_ERROR(err);

  return Status::OK();
}

void OpenCLExecutionProvider::RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) {
  // FIXME: Is it possible to use arena on OpenCL? cl_mem is opaque pointer in
  // OpenCL 1.2 and Shared Virtual Memory (SVM) is only available in OpenCL
  // 2.0, which still have limited support on a wide range of devices. Without
  // SVM we are unable to slice pre-allocated buffer, thus, unable to use it as
  // an arena.
  //
  // See https://stackoverflow.com/a/40951614
  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [=](int) {
        return std::make_unique<opencl::OpenCLBufferAllocator>(ctx_);
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [=](int) {
        return std::make_unique<opencl::OpenCLImage2DAllocator>(ctx_, UseFp16());
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }}));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUInputAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUInput));
      }}));
}

IAllocatorUniquePtrToClMem OpenCLExecutionProvider::GetScratchBuffer(size_t nbytes) const {
  ZoneScopedN("OpenCLExecutionProvider::GetScratchBuffer");
  auto alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_BUFFER);
  return {
      static_cast<cl_mem>(alloc->Alloc(nbytes)),
      [=](void* ptr) {
        alloc->Free(ptr);
      }};
}

IAllocatorUniquePtrToClMem OpenCLExecutionProvider::GetScratchImage2D(const opencl::Image2DDesc& desc) const {
  ZoneScopedN("OpenCLExecutionProvider::GetScratchImage2D");
  auto base_alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_IMAGE_2D);
  auto* alloc = static_cast<opencl::OpenCLImage2DAllocator*>(base_alloc.get());
  return IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>>{
      static_cast<cl_mem>(alloc->Alloc(desc)),
      [=](void* ptr) {
        alloc->Free(ptr);
      }};
}

Status OpenCLExecutionProvider::AfterCLLaunch() const {
  if (flush_after_launch_) {
    ORT_RETURN_IF_CL_ERROR(clFlush(cmd_queue_), "command queue flush failure.");
  }
  return Status::OK();
}

const opencl::OpenCLProgramManager* OpenCLExecutionProvider::GetProgramManager() const {
  return program_manager_.get();
}

opencl::OpenCLProgramManager* OpenCLExecutionProvider::GetProgramManager() {
  return program_manager_.get();
}

/*
#pragma region IDataTransfer related code
*/
std::unique_ptr<onnxruntime::IDataTransfer> OpenCLExecutionProvider::GetDataTransfer() const {
  return std::make_unique<opencl::OpenCLDataTransfer>(this, copy_kernels_.get());
}

namespace {
#define CONTENT_NAME copy_tensors_src
#include "opencl_generated/kernels/copy_tensors.cl.inc"
}  // namespace

void OpenCLExecutionProvider::InitCopyKernels() {
  ZoneScopedN("OpenCLExecutionProvider::InitCopyKernels");
  copy_kernels_ = std::make_unique<opencl::OpenCLKernelHolder>(GetProgramManager());
  copy_kernels_->LoadProgram(copy_tensors_src, copy_tensors_src_len);
  copy_kernels_->LoadKernel("CopyBuffer1DToImage2D");
  copy_kernels_->LoadKernel("CopyBuffer2DToImage2D");
  copy_kernels_->LoadKernel("CopyImage2DToBuffer1D");
  copy_kernels_->LoadKernel("CopyBufferNCHWToImage2D");
  copy_kernels_->LoadKernel("CopyImage2DToBufferNCHW");
}
/*
#pragma endregion
*/

bool OpenCLExecutionProvider::ShouldFlushAfterLaunch(const std::string& device_name) {
  return device_name.find("Mali") != std::string::npos;
}

}  // namespace onnxruntime
