#include "resize_image2d.h"

#include "core/providers/cpu/tensor/upsample.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME resize_kernel_src
#include "opencl_generated/tensor/kernels/resize_image2d.cl.inc"
}  // namespace

class Resize : public OpenCLKernel, UpsampleBase {
 public:
  explicit Resize(const OpKernelInfo& info)
      : OpenCLKernel(info), UpsampleBase(info) {
    VLOGS_DEFAULT(0) << "Init Resize (OpenCLKernel)";
    LoadProgram(resize_kernel_src, resize_kernel_src_len);
    LoadKernel("ResizeBilinear2D");
    LoadKernel("ResizeNearest2D");
  };

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("Resize::Compute");
    VLOG_CL_NODE();
    ORT_RETURN_IF(mode_ != UpsampleMode::LINEAR && mode_ != UpsampleMode::NN, "only supports linear interpolation and nearest interpolation");

    const auto* X = context->Input<Tensor>(0);
    const auto& X_shape = X->Shape();
    ORT_RETURN_IF(X_shape.NumDimensions() != 4, "only support 4D NCHW input");

    TensorShapeVector Y_shape(X->Shape().GetDims().size());
    if (scales_.empty()) {
      const auto* scales = context->Input<Tensor>(scales_input_idx_);
      const auto* sizes = context->Input<Tensor>(sizes_input_idx_);
      // Get scales data
      scales_.resize(X->Shape().GetDims().size());

      if (scales != nullptr && scales->Shape().Size() != 0) {
        ORT_ENFORCE(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
        ParseScalesData(scales, scales_);
        ComputeOutputShape(scales_, X_shape.GetDims(), Y_shape);
      } else {
        ORT_ENFORCE(sizes != nullptr && sizes->Shape().Size() != 0, "Either scales or sizes MUST be provided as input.");

        // When sizes input is available directly populate it into the output_dims array.
        memcpy(Y_shape.data(), sizes->Data<int64_t>(), sizes->Shape().Size() * sizeof(int64_t));

        ORT_ENFORCE(X->Shape().GetDims().size() == Y_shape.size(),
                    "Resize: input tensor's rank does not match the output tensor's rank.");
        ParseScalesDataFromOutputSize(Y_shape, X->Shape().GetDims(), scales_);
      }
    } else {
      ComputeOutputShape(scales_, X_shape.GetDims(), Y_shape);
    }

    const auto* Y = context->Output(0, Y_shape);
    VLOG_CL_IMAGE2D("Input", X);
    VLOG_CL_IMAGE2D("Output", Y);

    auto desc = Image2DDesc::PackFromTensorNCHW(Y->Shape());
    std::string kernel_name;
    if (mode_ == UpsampleMode::LINEAR) {
      kernel_name = "ResizeBilinear2D";
    } else if (mode_ == UpsampleMode::NN) {
      kernel_name = "ResizeNearest2D";
    }

    ZoneNamedN(_tracy_Resize, "Resize (kernel launch)", true);
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel(kernel_name)}
            .setInt2(desc.Width(), desc.Height())
            .setImage2Ds(*X, *Y)
            .setInt2(X_shape[3], X_shape[2])
            .setInt2(Y_shape[3], Y_shape[2])
            .setArg<cl_float>(1.0 / scales_[3])
            .setArg<cl_float>(1.0 / scales_[2])
            .setArg<cl_int>(coordinate_transform_mode_)
            .Launch(*exec_, desc.AsNDRange()));

    return Status::OK();
  }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    11, 12,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)
        .InputMemoryType(OrtMemTypeCPUInput, {1, 2, 3})
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),
    Resize)

ONNX_OPENCL_OPERATOR_KERNEL(
    Resize,
    13,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)
        .InputMemoryType(OrtMemTypeCPUInput, {1, 2, 3})
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),
    Resize)

}  // namespace opencl
}  // namespace onnxruntime
