#include "elementwise.h"

#include <sstream>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace {

#define CONTENT_NAME elementwise_kernel_src
#include "opencl_generated/math/kernels/elementwise.cl.inc"

std::string GetKernelSrc(const std::string& name_define, const std::string& type_define, const std::string& op_define) {
  std::stringstream oss;
  oss << "#define NAME " << name_define << "\n"
      << "#define T " << type_define << "\n"
      << "#define OP(X,Y) " << op_define << "\n"
      << elementwise_kernel_src;
  return oss.str();
}

}  // namespace

namespace onnxruntime {
namespace opencl {

#define ELEMENT_WISE_OP_IMPL(CLASS_NAME, OP_DEFINE)                                     \
  class CLASS_NAME : public OpenCLKernel {                                              \
   public:                                                                              \
    explicit CLASS_NAME(const OpKernelInfo& info) : OpenCLKernel(info) {                \
      VLOGS_DEFAULT(0) << "Init " #CLASS_NAME " (OpenCLKernel)";                        \
      LoadProgram(GetKernelSrc((#CLASS_NAME), "float", (OP_DEFINE)));                   \
      LoadKernel(#CLASS_NAME);                                                          \
    };                                                                                  \
                                                                                        \
    Status Compute(OpKernelContext* context) const override {                           \
      VLOGS_DEFAULT(0) << "[CL] Node: " << context->GetNodeName()                       \
                       << ", num inputs: " << context->InputCount()                     \
                       << ", num outputs: " << context->OutputCount() << std::endl;     \
      const auto* a = context->Input<Tensor>(0);                                        \
      const auto* b = context->Input<Tensor>(1);                                        \
      const auto* c = context->Output(0, a->Shape());                                   \
      VLOGS_DEFAULT(0) << "[CL]  Input[0] shape " << a->Shape() << " " << a->DataRaw()  \
                       << "--> cl::Buffer(" << CL_BUFFER_FROM_TENSOR(*a)() << ")";      \
      VLOGS_DEFAULT(0) << "[CL]  Input[1] shape " << b->Shape() << " " << b->DataRaw()  \
                       << "--> cl::Buffer(" << CL_BUFFER_FROM_TENSOR(*b)() << ")";      \
      VLOGS_DEFAULT(0) << "[CL]  Output[0] shape " << c->Shape() << " " << c->DataRaw() \
                       << "--> cl::Buffer(" << CL_BUFFER_FROM_TENSOR(*c)() << ")";      \
                                                                                        \
      size_t n = a->Shape().Size();                                                     \
                                                                                        \
      auto kernel = GetKernel(#CLASS_NAME);                                             \
      ORT_RETURN_IF_CL_ERROR(kernel.setArg(0, CL_BUFFER_FROM_TENSOR(*a)));              \
      ORT_RETURN_IF_CL_ERROR(kernel.setArg(1, CL_BUFFER_FROM_TENSOR(*b)));              \
      ORT_RETURN_IF_CL_ERROR(kernel.setArg(2, CL_BUFFER_FROM_TENSOR(*c)));              \
      ORT_RETURN_IF_CL_ERROR(kernel.setArg<cl_int>(3, n));                              \
      ORT_RETURN_IF_CL_ERROR(GetCommandQueue().enqueueNDRangeKernel(kernel, {0}, {n})); \
      return Status::OK();                                                              \
    }                                                                                   \
  };                                                                                    \
                                                                                        \
  ONNX_OPENCL_OPERATOR_KERNEL(                                                          \
      CLASS_NAME,                                                                       \
      7,                                                                                \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),     \
      CLASS_NAME)

// FIXME: disable temporarily, for testing image2d kernel
// ELEMENT_WISE_OP_IMPL(Add, "(X)+(Y)");
// ELEMENT_WISE_OP_IMPL(Sub, "(X)-(Y)");
// ELEMENT_WISE_OP_IMPL(Mul, "(X)*(Y)");
// ELEMENT_WISE_OP_IMPL(Div, "(X)/(Y)");

}  // namespace opencl
}  // namespace onnxruntime
