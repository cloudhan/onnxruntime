// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_execution_provider.h"
#include "opencl_allocator.h"
#include "opencl_kernel_holder.h"
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
#include "core/providers/opencl/nn/conv_image2d.h"

namespace onnxruntime {
namespace opencl {

Status RegisterOpenCLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 6, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, Conv)>,
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
  ORT_THROW_IF_ERROR(InitOpenCLContext());
}

OpenCLExecutionProvider::OpenCLExecutionProvider(OpenCLExecutionProvider&& provider) noexcept
    : IExecutionProvider(kOpenCLExecutionProvider), use_fp16_(provider.use_fp16_) {
  std::swap(dev_, provider.dev_);
  std::swap(ctx_, provider.ctx_);
  std::swap(cmd_queue_, provider.cmd_queue_);
}

OpenCLExecutionProvider::~OpenCLExecutionProvider() = default;

std::shared_ptr<KernelRegistry> OpenCLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = opencl::GetOpenCLKernelRegistry();
  return kernel_registry;
}

Status OpenCLExecutionProvider::InitOpenCLContext() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  VLOGS_DEFAULT(1) << "[CL] num platforms: " << platforms.size();
  ORT_ENFORCE(!platforms.empty());
  // FIXME: add platform selection logic
  cl::Platform selected_platform = platforms[0];
  for (const auto& plat : platforms) {
    auto vendor = plat.getInfo<CL_PLATFORM_VENDOR>();
    std::cout << "[CL] platform vendor: " << vendor << "\n";
    if (vendor == "Oclgrind") {
      selected_platform = plat;
      break;
    }
  }
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(selected_platform)(), 0};
  {
    cl_int err{};
    ctx_ = cl::Context(CL_DEVICE_TYPE_GPU, properties, /*notifyFptr=*/nullptr, /*data=*/nullptr, &err);
    ORT_RETURN_IF_CL_ERROR(err);
  }

  std::vector<cl::Device> devices = ctx_.getInfo<CL_CONTEXT_DEVICES>();
  VLOGS_DEFAULT(1) << "[CL] num devices: " << devices.size();
  ORT_ENFORCE(!devices.empty());
  // FIXME: add device selection logic
  dev_ = std::move(devices[0]);

  // NOTE: use stdout for mobile
  std::cout << "[CL] device name: " << dev_.getInfo<CL_DEVICE_NAME>() << "\n";
  std::cout << "[CL] device vendor: " << dev_.getInfo<CL_DEVICE_VENDOR>() << "\n";
  std::cout << "[CL] device version: " << dev_.getInfo<CL_DEVICE_VERSION>() << "\n";
  auto exts = dev_.getInfo<CL_DEVICE_EXTENSIONS>();
  std::cout << "[CL] device extensions: " << exts << std::endl;
  bool has_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
  if (!has_fp16 && UseFp16()) {
    LOGS_DEFAULT(ERROR) << "[CL] FP16 is requested, but is not supported by the device!";
    DisableFp16();
  }
  LOGS_DEFAULT(INFO) << "[CL] FP16: " << UseFp16();

  {
    cl_int err{};
    cmd_queue_ = cl::CommandQueue(ctx_, dev_, /*properties=*/0, &err);
    ORT_RETURN_IF_CL_ERROR(err);
  }

  InitCopyKernels();

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
        return std::make_unique<opencl::OpenCLBufferAllocator>(this->ctx_);
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [=](int) {
        return std::make_unique<opencl::OpenCLImage2DAllocator>(this->ctx_, this->UseFp16());
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }}));
}

IAllocatorUniquePtr<cl::Buffer> OpenCLExecutionProvider::GetScratchBuffer(size_t nbytes) const {
  auto alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_BUFFER);
  return IAllocatorUniquePtr<cl::Buffer>{
      static_cast<cl::Buffer*>(alloc->Alloc(nbytes)),
      [=](void* ptr) {
        alloc->Free(ptr);
      }};
}

IAllocatorUniquePtr<cl::Image2D> OpenCLExecutionProvider::GetScratchImage2D(opencl::Image2DDesc desc) const {
  auto alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_IMAGE_2D);
  return IAllocatorUniquePtr<cl::Image2D>{
      static_cast<cl::Image2D*>(alloc->Alloc(desc.AsTensorShape())),
      [=](void* ptr) {
        alloc->Free(ptr);
      }};
}

/*
#pragma region IDataTransfer related code
*/
std::unique_ptr<onnxruntime::IDataTransfer> OpenCLExecutionProvider::GetDataTransfer() const {
  return std::make_unique<opencl::OpenCLDataTransfer>(this, copy_kernels_.get());
}

namespace {
#define CONTENT_NAME copy1d_kernel_src
#include "opencl_generated/kernels/copy_tensor_1d.cl.inc"
#define CONTENT_NAME copynchw_kernel_src
#include "opencl_generated/kernels/copy_tensor_nchw.cl.inc"
#define CONTENT_NAME copynchw4_kernel_src
#include "opencl_generated/kernels/copy_tensor_nchw4.cl.inc"
#define CONTENT_NAME copyconv2d_kernel_src
#include "opencl_generated/kernels/copy_conv2d_weight.cl.inc"
#define CONTENT_NAME copydepthwiseconv2d_kernel_src
#include "opencl_generated/kernels/copy_depthwise_conv2d_weight.cl.inc"
}  // namespace

void OpenCLExecutionProvider::InitCopyKernels() {
  std::ostringstream oss;
  oss << std::string(copy1d_kernel_src, copy1d_kernel_src_len) << "\n";
  oss << std::string(copynchw_kernel_src, copynchw_kernel_src_len) << "\n";
  oss << std::string(copynchw4_kernel_src, copynchw4_kernel_src_len) << "\n";
  oss << std::string(copyconv2d_kernel_src, copyconv2d_kernel_src_len) << "\n";
  oss << std::string(copydepthwiseconv2d_kernel_src, copydepthwiseconv2d_kernel_src_len) << "\n";
  copy_kernels_ = std::make_unique<opencl::OpenCLKernelHolder>();
  copy_kernels_->LoadProgram(this, oss.str());
  copy_kernels_->LoadKernel("CopyBuffer1DToImage2D");
  copy_kernels_->LoadKernel("CopyImage2DToBuffer1D");
  copy_kernels_->LoadKernel("CopyBufferNCHWToImage2D");
  copy_kernels_->LoadKernel("CopyImage2DToBufferNCHW");
  copy_kernels_->LoadKernel("Conv2DWeightBufferToImage");
  copy_kernels_->LoadKernel("CopyDepthwiseConvWeightBufferToImage");
}
/*
#pragma endregion
*/

}  // namespace onnxruntime
