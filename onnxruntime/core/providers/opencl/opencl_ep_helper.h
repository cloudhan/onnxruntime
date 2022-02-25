#include "opencl_forward_decl.h"

namespace onnxruntime {
namespace opencl {

class SupportedNodeHelper {
  using KCIInfo = std::unordered_map<std::string, std::vector<const KernelCreateInfo*>>;

 public:
  SupportedNodeHelper(const KernelRegistry* kernel_registry, const logging::Logger* logger);

  bool IsNodeSupported(const Node& node);
  std::unordered_set<const Node*> GetSupportedNodes(const GraphViewer& graph_viewer);

 private:
  bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape);

  bool IsInputSupported(const NodeArg& node_arg, const std::string& node_name, const KernelCreateInfo* kci);
  bool IsInputsSupported(const Node& node, const KernelCreateInfo* kci);

  KCIInfo BuildOpTypeToKernelDefMapping(const KernelRegistry* kernel_registry);

  const KernelRegistry* kernel_registry_;
  KCIInfo op_name_to_kci_;
  const logging::Logger* logger_;
};

}  // namespace opencl
}  // namespace onnxruntime
