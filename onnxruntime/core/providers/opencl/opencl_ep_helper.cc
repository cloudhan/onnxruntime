#include "opencl_ep_helper.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace opencl {

SupportedNodeHelper::SupportedNodeHelper(const KernelRegistry* kernel_registry, const logging::Logger* logger)
    : op_name_to_kci_{BuildOpTypeToKernelDefMapping(kernel_registry)}, logger_{logger} {
  ;
}

SupportedNodeHelper::KCIInfo SupportedNodeHelper::BuildOpTypeToKernelDefMapping(const KernelRegistry* kernel_registry) {
  std::unordered_map<std::string, std::vector<const KernelCreateInfo*>> ret;
  for (auto& [_, hash_value] : kernel_registry->ExportKernelDefHashes()) {
    const KernelCreateInfo* kci;
    if (!kernel_registry->TryFindKernelByHash(hash_value, &kci)) {
      ORT_THROW("BuildOpTypeToKernelDefMapping Fatal Error.");
    }
    ret[kci->kernel_def->OpName()].push_back(kci);
  }
  return ret;
}

bool SupportedNodeHelper::GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape) {
  const auto* shape_proto = node_arg.Shape();
  if (!shape_proto) {
    LOGS(*logger_, WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  // We already checked the shape has no dynamic dimension
  for (const auto& dim : shape_proto->dim()) {
    shape.push_back(dim.dim_value());
  }

  return true;
}

bool SupportedNodeHelper::IsInputSupported(const NodeArg& input,
                                           const std::string& node_name,
                                           const KernelCreateInfo* kci) {
  // TODO: we should also check shape in this function, because the Image2D has size limit.
  const auto& input_name = input.Name();
  const auto* shape_proto = input.Shape();
  if (shape_proto == nullptr) {
    LOGS(*logger_, VERBOSE) << "OpenCL EP does not support Node [" << node_name << "]. "
                            << "Reason: Its Input [" << input_name << "] does not have a shape.";
    return false;
  }

  // FIXME: How to typecheck?
  /*const auto* param_type_proto = input.TypeAsProto();
  if (param_type_proto == nullptr || !param_type_proto->has_tensor_type()) {
    LOGS(*logger_, VERBOSE) << "OpenCL EP does not support Node [" << node_name << "]. "
                            << "Reason: Its Input [" << input_name << "] does not have a type.";
    return false;
  }

  const auto& tc = kci->kernel_def->EnabledTypeConstraints();
  auto supported_types_iter = tc.find(input_name);
  if (supported_types_iter == tc.end()) {
    LOGS(*logger_, VERBOSE) << "OpenCL EP does not support Node [" << node_name << "]. "
                            << "Reason: Its Input [" << input_name << "] is not typed.";
    return false;
  }

  if (!std::any_of(supported_types_iter->second.cbegin(), supported_types_iter->second.cend(),
                   [&](const DataTypeImpl* formal) { return formal->IsCompatible(*param_type_proto); })) {
    LOGS(*logger_, VERBOSE) << "OpenCL EP does not support Node [" << node_name << "]. "
                            << "Reason: Its Input [" << input_name << "] type is not supported.";
    return false;
  }*/
  return true;
}

bool SupportedNodeHelper::IsInputsSupported(const Node& node, const KernelCreateInfo* kci) {
  auto node_name = node.Name();
  const auto& inputs = node.InputDefs();
  return !std::any_of(inputs.cbegin(), inputs.cend(),
                     [&](const NodeArg* input) { return !IsInputSupported(*input, node_name, kci); });
}

bool SupportedNodeHelper::IsNodeSupported(const Node& node) {
  auto op_name = node.OpType();
  auto it = op_name_to_kci_.find(op_name);
  if (it == op_name_to_kci_.end()) {
    LOGS(*logger_, VERBOSE) << "OpenCL EP does not support Node [" << node.Name() << "]. "
                            << "Reason: Op [" << op_name << "] is not supported.";
    return false;
  }

  auto supported_kci_it = std::find_if(it->second.cbegin(), it->second.cend(), [&](const KernelCreateInfo* kci) {
    int start = std::numeric_limits<int>::max();
    int end = std::numeric_limits<int>::min();
    kci->kernel_def->SinceVersion(&start, &end);
    return node.SinceVersion() >= start && node.SinceVersion() <= end;
  });
  if (supported_kci_it == it->second.cend()) {
    LOGS(*logger_, VERBOSE) << "OpenCL EP does not support Node [" << node.Name() << "]. "
                            << "Reason: Op [" << op_name << "] version " << node.SinceVersion() << "is not supported";
    return false;
  }

  return IsInputsSupported(node, *supported_kci_it);
}

std::unordered_set<const Node*> SupportedNodeHelper::GetSupportedNodes(const GraphViewer& graph_viewer) {
  std::unordered_set<const Node*> supported_nodes{};

  for (const auto& node : graph_viewer.Nodes()) {
    const bool supported = IsNodeSupported(node);
    VLOGS(*logger_, 1) << "Operator type: [" << node.OpType()
                       << "] index: [" << node.Index()
                       << "] name: [" << node.Name()
                       << "] supported: [" << supported
                       << "]";
    if (supported) {
      supported_nodes.insert(&node);
    }
  }

  return supported_nodes;
}

}  // namespace opencl
}  // namespace onnxruntime
