// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/annotate_node_arg.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/tensor_usage.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {

AnnotateNodeArg::AnnotateNodeArg(const KernelRegistryManager& registry_manager) noexcept
    : GraphTransformer("AnnotateNodeArg", {kOpenCLExecutionProvider}),
      registry_manager_{registry_manager} {
}

namespace {

TensorUsage ClassifyTensorUsage(const Node* node, const NodeArg* node_arg, int node_arg_index, const KernelRegistryManager& registry_mgr) {
  const KernelCreateInfo* kci;
  ORT_THROW_IF_ERROR(registry_mgr.SearchKernelRegistry(*node, &kci));
  const auto& name = kci->kernel_def->OpName();
  if ((name == "Conv" || name == "FusedConv")) {
    if (node_arg_index != 1) { // the node_arg is W
      return TensorUsage::Generic;
    }

    // Get group attribute
    const auto& attrs = node->GetAttributes();
    auto it = attrs.find("group");
    ORT_ENFORCE(it != attrs.end());
    auto group = it->second.i();

    const auto* strides_attr = graph_utils::GetNodeAttribute(*node, "strides");
    const auto* dilations_attr = graph_utils::GetNodeAttribute(*node, "dilations");
    ORT_ENFORCE(strides_attr!=nullptr);
    ORT_ENFORCE(dilations_attr != nullptr);
    auto strides = RetrieveValues<int64_t>(*strides_attr);
    auto dilations = RetrieveValues<int64_t>(*dilations_attr);

    // get number of output channel
    auto dim_channel_out = node_arg->Shape()->dim(0);
    ORT_ENFORCE(utils::HasDimValue(dim_channel_out));
    auto co_total = dim_channel_out.dim_value();
    auto co_per_group = co_total / group;

    // get number of input channel
    auto dim_channel_in = node_arg->Shape()->dim(1);
    ORT_ENFORCE(utils::HasDimValue(dim_channel_in));
    auto ci_per_group = dim_channel_in.dim_value();

    // get number of input channel
    auto dim_kernel_h = node_arg->Shape()->dim(2);
    ORT_ENFORCE(utils::HasDimValue(dim_kernel_h));
    auto kernel_h = dim_kernel_h.dim_value();

    // get number of input channel
    auto dim_kernel_w = node_arg->Shape()->dim(3);
    ORT_ENFORCE(utils::HasDimValue(dim_kernel_w));
    auto kernel_w = dim_kernel_w.dim_value();

    auto input_arg = node->InputDefs()[0];
    auto dim_in_heigth = input_arg->Shape()->dim(3);
    ORT_ENFORCE(utils::HasDimValue(dim_in_heigth));
    auto in_heigth = dim_in_heigth.dim_value();


    if (ci_per_group == 1 && co_per_group == 1) {
      // TODO: relax co_per_group requirement
      return TensorUsage::DepthwiseConvWeight;
    } else if (kernel_w == 3 && kernel_h == 3 && strides.size() == 2 
        && strides[0] == 1 && strides[1] == 1 && strides.size() == 2 
        && strides[0] == 1 && strides[1] == 1 && co_total >= 32 && 
        ci_per_group >= 32 && (in_heigth * 1.0 / ci_per_group) <= 4) {
      return TensorUsage::WinogradWeight;
    }
    return TensorUsage::ConvWeight;
  }

  return TensorUsage::Generic;
}

// FIXME: use GetIndexFromName
size_t GetInputIndex(const Node* node, const NodeArg* node_arg) {
  auto arg_it = std::find_if(node->InputDefs().begin(), node->InputDefs().end(),
                             [&](const NodeArg* o) { return o->Name() == node_arg->Name(); });
  ORT_ENFORCE(arg_it != node->InputDefs().end());
  return std::distance(node->InputDefs().begin(), arg_it);
}

}  // namespace

Status AnnotateNodeArg::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const KernelCreateInfo* kci;

  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* node_ptr = graph.GetNode(node_index);
    ORT_RETURN_IF(node_ptr == nullptr, "node must not be nullptr");
    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(*node_ptr, modified, graph_level, logger));

    // Annotate for tensor memory type, the DataTransfer::CopyTensor is unable
    // to handle the case where the an ExecutionProvider's allocator have
    // multiple memory type. For example, Buffer and Image2D in OpenCL.
    ORT_RETURN_IF_ERROR(registry_manager_.get().SearchKernelRegistry(node, &kci));
    if (kci->kernel_def->OpName() == "MemcpyFromHost") {
      auto outs = node.MutableOutputDefs();
      ORT_RETURN_IF(outs.size() != 1, "MemcpyFromHost must have 1 and only 1 output");
      NodeArg* copy_out = outs[0];

      // The result of MemcpyFromHost will be consumed by other nodes, those
      // nodes's input may or may not have InputMemoryType specified on
      // KernelDefBuilder. Since it is MemcpyFromHost, the memory resides on
      // device. The MemoryType should remain the same and live on same device
      // type.
      auto copy_out_consumers = graph_viewer.GetConsumerNodes(copy_out->Name());
      for (const auto* consumer : copy_out_consumers) {
        auto arg_idx = GetInputIndex(consumer, copy_out);
        ORT_RETURN_IF_ERROR(registry_manager_.get().SearchKernelRegistry(*consumer, &kci));
        auto mem_type = kci->kernel_def->InputMemoryType(arg_idx);
        if (copy_out->HasMemoryType()) {
          ORT_RETURN_IF(copy_out->MemoryType() != mem_type, "dst memory type is different, ill-formed MemcpyFromHost");
        } else {
          copy_out->SetMemoryType(mem_type);
        }
      }
    }
  }

  // Annotate for tensor usage. Tensor is layout optimized in some case. So the
  // DataTransfer::CopyTensor should have the additional information.
  auto tensors = graph_viewer.GetAllInitializedTensors();
  for (auto& [arg_name, tensor_proto] : tensors) {
    auto* node_arg = graph.GetNodeArg(arg_name);
    auto nodes = graph_viewer.GetConsumerNodes(arg_name);
    for (auto& node : nodes) {
      auto index = GetInputIndex(node, node_arg);
      auto usage = ClassifyTensorUsage(node, node_arg, index, registry_manager_);
      if (node_arg->HasUsage()) {
        ORT_RETURN_IF(node_arg->Usage() != usage, "Ill-formed tensor Usage");
      } else {
        node_arg->SetUsage(usage);
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
