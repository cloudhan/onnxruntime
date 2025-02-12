#include "dnnl_binary.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlBinary::DnnlBinary() {}

void DnnlBinary::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto eng = sp.GetEngine();

  dnnl::algorithm algo;
  if (node.OpType() == "Add") {
    algo = dnnl::algorithm::binary_add;
  } else if (node.OpType() == "Mul") {
    algo = dnnl::algorithm::binary_mul;
  } else if (node.OpType() == "Sub") {
    algo = dnnl::algorithm::binary_sub;
  } else if (node.OpType() == "Div") {
    algo = dnnl::algorithm::binary_div;
  } else {
    ORT_THROW("op type not supported");
  }

  auto src_0_ori_md = sp.GetMemory(node.Input(IN_A)).get_desc();
  auto src_1_ori_md = sp.GetMemory(node.Input(IN_B)).get_desc();

  auto src_0_dims = src_0_ori_md.dims();
  auto src_1_dims = src_1_ori_md.dims();
  if (src_0_dims.size() != src_1_dims.size()) {
    while (src_0_dims.size() < src_1_dims.size()) {
      src_0_dims.insert(src_0_dims.begin(), 1);
    }
    while (src_0_dims.size() > src_1_dims.size()) {
      src_1_dims.insert(src_1_dims.begin(), 1);
    }
  }

  auto src_0_md = src_0_ori_md.reshape(src_0_dims);
  auto src_1_md = src_1_ori_md.reshape(src_1_dims);

  auto output_shape = src_0_dims;
  for (size_t i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] == 1) {
      output_shape[i] = src_1_dims[i];
    }
  }

  auto dst_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), dnnl::memory::format_tag::any);

  auto binary_d = dnnl::binary::desc(algo, src_0_md, src_1_md, dst_md);
  auto binary_pd = dnnl::binary::primitive_desc(binary_d, eng);

  auto binary_src0_mem = sp.GetMemoryAndReshape(node.Input(IN_A), binary_pd.src0_desc(), eng);
  auto binary_src1_mem = sp.GetMemoryAndReshape(node.Input(IN_B), binary_pd.src1_desc(), eng);

  auto binary_dst_mem = dnnl::memory(binary_pd.dst_desc(), eng);
  auto binary_prim = dnnl::binary(binary_pd);

  sp.AddPrimitive(binary_prim, {{DNNL_ARG_SRC_0, binary_src0_mem},
                                {DNNL_ARG_SRC_1, binary_src1_mem},
                                {DNNL_ARG_DST, binary_dst_mem}});

  sp.SetMemory(node.Output(OUT_Y), binary_dst_mem);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
