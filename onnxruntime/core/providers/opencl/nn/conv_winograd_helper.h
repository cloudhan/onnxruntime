#ifndef WINOGRAD_Helper_H_
#define WINOGRAD_Helper_H_

#include <memory>
#include <tuple>
#include "core/framework/tensor.h"

namespace onnxruntime {
typedef std::vector<int64_t> FBshape;
struct DirectBuffer;
using DirectBuffer_ptr = std::shared_ptr<DirectBuffer>;

struct DirectBuffer {
  FBshape shape;
  int64_t size;
  std::shared_ptr<float[]> buff;
  ~DirectBuffer() {
  }
  DirectBuffer() : size(0) {

  }
  DirectBuffer(const DirectBuffer& other) = delete;
  DirectBuffer(DirectBuffer&& other) = delete;

  //pass value here won't impact  performance
  void create(FBshape shape_) {
    shape = shape_;
    size = 1;
    for (auto n : shape) {
      size *= n;
    }
    buff = std::shared_ptr<float[]>(new float[size], [](float* p) { delete[] p; });
    return;
  }
  void create(int64_t w, int64_t h) {
    create({w, h});
  }
  int fill(std::vector<float> v) {
    if (v.size() != size) {
      return -1;
    }
    std::memcpy(buff.get(), v.data(), size*sizeof(float));
    return 0;
  }
};

class WinogradHelper {
 public:
  WinogradHelper(int computeUnit, int kernelSize);
  ~WinogradHelper() = default;
    
  DirectBuffer_ptr transformWeight(const float* source, int output_channel, int input_channel);

 private:
  DirectBuffer_ptr allocWeightTensor(int batch, int channel, int unitCi, int unitCo);

 private:
  DirectBuffer_ptr G_;
  int wino_size_;
  int unit_;
  int kernel_size_;
};

}  // namespace onnxruntime

#endif  //WINOGRAD_Helper_H_
