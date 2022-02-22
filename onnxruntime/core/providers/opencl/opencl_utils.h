#pragma once

#include <CL/opencl.hpp>

#include <cstdio>
#include <iomanip>
#include <sstream>
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "opencl_forward_decl.h"
#include "opencl_execution_provider.h"

#ifdef NDEBUG
#define USE_CL_CHECKED_CAST 0
#else
#define USE_CL_CHECKED_CAST 1
#endif

#define ONNX_OPENCL_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kOpenCLExecutionProvider, builder, __VA_ARGS__)

#define OPENCL_EXEC_PROVIDER_FROM_INFO(info) \
  const_cast<OpenCLExecutionProvider*>(static_cast<const OpenCLExecutionProvider*>((info).GetExecutionProvider()))

#define TO_STRING_(T) #T
#define TO_STRING(T) TO_STRING_(T)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ORT_RETURN_IF_CL_ERROR(error_code, ...)                                          \
  if ((error_code) != CL_SUCCESS) {                                                      \
    std::ostringstream oss;                                                              \
    oss << __FILE__ ":" TO_STRING(__LINE__)                                              \
        << "\nOpenCL Error Code  : " << (int)(error_code)                                \
        << "\n       Error String: " << onnxruntime::opencl::GetErrorString(error_code); \
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, oss.str(),                              \
                           ::onnxruntime::MakeString(__VA_ARGS__));                      \
  }

#define ORT_THROW_IF_CL_ERROR(error_code, ...)                                           \
  if ((error_code) != CL_SUCCESS) {                                                      \
    std::ostringstream oss;                                                              \
    oss << __FILE__ ":" TO_STRING(__LINE__)                                              \
        << "\nOpenCL Error Code  : " << (int)(error_code)                                \
        << "\n       Error String: " << onnxruntime::opencl::GetErrorString(error_code); \
    ORT_THROW(oss.str(), ::onnxruntime::MakeString(__VA_ARGS__));                        \
  }

#if USE_CL_CHECKED_CAST
#define CL_BUFFER_FROM_TENSOR(TENSOR) [&]() {                                              \
  auto* ptr = const_cast<cl::Buffer*>(static_cast<const cl::Buffer*>((TENSOR).DataRaw())); \
  cl_mem_object_type ret;                                                                  \
  ORT_THROW_IF_CL_ERROR(ptr->getInfo(CL_MEM_TYPE, &ret));                                  \
  ORT_ENFORCE(ret == CL_MEM_OBJECT_BUFFER, ptr, " is not cl::Buffer");                     \
  return *ptr;                                                                             \
}()

#define CL_IMAGE2D_FROM_TENSOR(TENSOR) [&]() {                                               \
  auto* ptr = const_cast<cl::Image2D*>(static_cast<const cl::Image2D*>((TENSOR).DataRaw())); \
  cl_mem_object_type ret;                                                                    \
  ORT_THROW_IF_CL_ERROR(ptr->getInfo(CL_MEM_TYPE, &ret));                                    \
  ORT_ENFORCE(ret == CL_MEM_OBJECT_IMAGE2D, ptr, " is not cl::Image2D");                     \
  return *ptr;                                                                               \
}()
#else
#define CL_BUFFER_FROM_TENSOR(TENSOR) (*const_cast<cl::Buffer*>(static_cast<const cl::Buffer*>((TENSOR).DataRaw())))
#define CL_IMAGE2D_FROM_TENSOR(TENSOR) (*const_cast<cl::Image2D*>(static_cast<const cl::Image2D*>((TENSOR).DataRaw())))
#endif

#define VLOG_CL_NODE()                                          \
  VLOGS_DEFAULT(0) << "[CL] Node: " << context->GetNodeName()   \
                   << ", num inputs: " << context->InputCount() \
                   << ", num outputs: " << context->OutputCount()
#define VLOG_CL_BUFFER(desc, tensor_ptr)                                      \
  VLOGS_DEFAULT(0) << "[CL]  " << std::setfill(' ') << std::setw(9) << (desc) \
                   << " shape " << (tensor_ptr)->Shape()                      \
                   << " " << (tensor_ptr)->DataRaw() << " --> cl::Buffer("    \
                   << CL_BUFFER_FROM_TENSOR(*(tensor_ptr))() << ")"
#define VLOG_CL_IMAGE2D(desc, tensor_ptr)                                     \
  VLOGS_DEFAULT(0) << "[CL]  " << std::setfill(' ') << std::setw(9) << (desc) \
                   << " shape " << (tensor_ptr)->Shape()                      \
                   << " " << (tensor_ptr)->DataRaw() << " --> cl::Image2D("   \
                   << CL_IMAGE2D_FROM_TENSOR(*(tensor_ptr))() << ")"

namespace onnxruntime {
namespace opencl {

inline std::string ToString(const cl::NDRange& range) {
  if (range.dimensions() == 0) {
    return "[<unspecified>]";
  } else if (range.dimensions() == 1) {
    return onnxruntime::MakeString("[", range[0], "]");
  } else if (range.dimensions() == 2) {
    return onnxruntime::MakeString("[", range[0], ",", range[1], "]");
  }
  return onnxruntime::MakeString("[", range[0], ",", range[1], ",", range[2], "]");
}

const char* GetErrorString(cl_int error_code);
cl::Program LoadProgram(const cl::Context& ctx, const cl::Device& dev, const std::string& src, bool use_fp16);
cl::Program LoadProgram(const cl::Context& ctx, const cl::Device& dev, const char* src, size_t src_len, bool use_fp16);
cl::Kernel LoadKernel(const cl::Program& program, const char* name);

// NOTE: for OrtDevice ctor
struct CLMemType {
  static constexpr OrtDevice::MemoryType OPENCL_BUFFER = OrtDevice::MemType::DEFAULT;
  static constexpr OrtDevice::MemoryType OPENCL_IMAGE_2D = 5;
};

// NOTE: for opencl internal definition.
enum MemoryKind : uint8_t {
  Buffer = CLMemType::OPENCL_BUFFER,
  Image2D = CLMemType::OPENCL_IMAGE_2D,
};

template <typename T, typename E = std::enable_if_t<std::is_integral_v<T>>>
T CeilDiv(T a, T b) {
  return (a - 1) / b + 1;
}

template <typename T1, typename T2, typename E = std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>>>
T1 CeilDiv(T1 a, T2 b) {
  return (a - 1) / b + 1;
}

template <typename T1, typename T2, typename E = std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>>>
T1 RoundToMultiple(T1 a, T2 m) {
  return CeilDiv(a, m) * m;
}

class Image2DDesc : private std::pair<int64_t, int64_t> {
 public:
  using pair::pair;

  static Image2DDesc PackFromTensor(const TensorShape& shape) {
    switch (shape.NumDimensions()) {
      case 1:
        return PackFromTensor1D(shape);
      case 2:
        return PackFromTensor2D(shape);
      case 4:
        return PackFromTensorNCHW(shape);
      case 5:
        return PackFromTensorNCHWc(shape);
      default:
        return {0, 0};
    }
  }

  static Image2DDesc PackFromTensor1D(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 1);
    //
    return {1024, CeilDiv(shape[0], 4 * 1024)};
  }

  static Image2DDesc PackFromTensor2D(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 2);
    return {CeilDiv(shape[0], 4), shape[1]};
  }

  static Image2DDesc PackFromTensorNCHW(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 4);
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t c = 4;
    int64_t Cc = CeilDiv(C, c);
    return {Cc * W, N * H};
  }

  // NCHWc is actually Tensor of shape N[C/c]HWc then packed as NH C/cWc
  static Image2DDesc PackFromTensorNCHWc(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 5);
    int64_t N = shape[0];
    int64_t Cc = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t c = shape[4];
    ORT_ENFORCE(c == 4);
    return {Cc * W, N * H};
  }

  static Image2DDesc PackFromConv2DWeight(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 4);
    int64_t C_o = shape[0];
    int64_t C_i = shape[1];
    int64_t K_h = shape[2];
    int64_t K_w = shape[3];
    return {C_i, CeilDiv(C_o, 4) * K_h * K_w};
  }

  static Image2DDesc PackFromDepthwiseConv2DWeight(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 4);
    int64_t C_o = shape[0];
    int64_t C_i = shape[1];
    int64_t K_h = shape[2];
    int64_t K_w = shape[3];
    return {K_h * K_w * C_i, CeilDiv(C_o, 4)};
  }

  auto Height() const {
    return second;
  }

  auto Width() const {
    return first;
  }

  size_t UHeight() const {
    return static_cast<size_t>(second);
  }

  size_t UWidth() const {
    return static_cast<size_t>(first);
  }

  TensorShape AsTensorShape() const {
    return {Width(), Height()};
  }

  cl::NDRange AsNDRange() const {
    return {UWidth(), UHeight()};
  }
};

// cl::make_kernel returns typed functor object. The problem is the type
// signature varys as the kernel signature changes, makes it cannot be stored
// in a cached kernel registry. So I choose to store cl::Kernel object and wrap
// it with a simpler form without typing issue.
class KernelLauncher {
 public:
  explicit KernelLauncher(const cl::Kernel& kernel) : kernel_{kernel}, index_{0}, err_{CL_SUCCESS} {}
  const cl::Kernel& Kernel() const { return kernel_; }

#define SKIP_IF_ERRORED(expr) \
  if (err_ == CL_SUCCESS) {   \
    err_ = (expr);            \
    err_index_ = index_;      \
  }

  template <typename T, typename E = std::is_convertible<T, cl_int>>
  KernelLauncher& setInt2(T v1, T v2) {
    cl_int tmp[2] = {static_cast<cl_int>(v1), static_cast<cl_int>(v2)};
    SKIP_IF_ERRORED(kernel_.setArg(index_, sizeof(tmp), tmp));
    index_ += 1;
    return *this;
  }

  template <typename T, typename E = std::is_convertible<T, cl_int>>
  KernelLauncher& setInt3(T v1, T v2, T v3) {
    cl_int tmp[3] = {static_cast<cl_int>(v1), static_cast<cl_int>(v2), static_cast<cl_int>(v3)};
    SKIP_IF_ERRORED(kernel_.setArg(index_, sizeof(tmp), tmp));
    index_ += 1;
    return *this;
  }

  template <typename T, typename E = std::is_convertible<T, cl_int>>
  KernelLauncher& setInt4(T v1, T v2, T v3, T v4) {
    cl_int tmp[4] = {static_cast<cl_int>(v1), static_cast<cl_int>(v2), static_cast<cl_int>(v3), static_cast<cl_int>(v4)};
    SKIP_IF_ERRORED(kernel_.setArg(index_, sizeof(tmp), tmp));
    index_ += 1;
    return *this;
  }

  template <typename T>
  KernelLauncher& setArg(const T& arg) {
    SKIP_IF_ERRORED(kernel_.setArg(index_, arg));
    index_ += 1;
    return *this;
  }

  KernelLauncher& setBuffer(const cl::Buffer& arg) {
    SKIP_IF_ERRORED(kernel_.setArg<cl::Buffer>(index_, arg));
    index_ += 1;
    return *this;
  }

  KernelLauncher& setBuffer(const Tensor& arg) {
    return setBuffer(CL_BUFFER_FROM_TENSOR(arg));
  }

  template <typename T1, typename T2>
  KernelLauncher& setBuffers(T1&& arg, T2&& other) {
    setBuffer(std::forward<T1>(arg));
    return setBuffer(std::forward<T2>(other));
  }

  template <typename T, typename... Ts>
  KernelLauncher& setBuffers(T&& arg, Ts&&... args) {
    setBuffer(std::forward<T>(arg));
    return setBuffers(std::forward<Ts>(args)...);
  }

  KernelLauncher& setImage2D(const cl::Image2D& arg) {
    SKIP_IF_ERRORED(kernel_.setArg<cl::Image2D>(index_, arg));
    index_ += 1;
    return *this;
  }

  KernelLauncher& setImage2D(const Tensor& arg) {
    return setImage2D(CL_IMAGE2D_FROM_TENSOR(arg));
  }

  template <typename T1, typename T2>
  KernelLauncher& setImage2Ds(T1&& arg, T2&& other) {
    setImage2D(std::forward<T1>(arg));
    return setImage2D(std::forward<T2>(other));
  }

  template <typename T, typename... Ts>
  KernelLauncher& setImage2Ds(T&& arg, Ts&&... args) {
    setImage2D(std::forward<T>(arg));
    return setImage2Ds(std::forward<Ts>(args)...);
  }

  Status Launch(const cl::CommandQueue& queue, const cl::NDRange& global, const cl::NDRange& local = cl::NullRange) {
    ORT_RETURN_IF_CL_ERROR(err_, " on setting argument ", static_cast<int>(err_index_));
    VLOGS_DEFAULT(1) << "[CL] Launching " << kernel_.getInfo<CL_KERNEL_FUNCTION_NAME>()
                     << " with global work size: " << ToString(global)
                     << " local work size: " << ToString(local);
    ORT_RETURN_IF_CL_ERROR(queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local));
    return Status::OK();
  }

#undef SKIP_IF_ERRORED

 private:
  cl::Kernel kernel_;
  cl_uint index_;
  cl_int err_;
  cl_uint err_index_;
};

}  // namespace opencl
}  // namespace onnxruntime
