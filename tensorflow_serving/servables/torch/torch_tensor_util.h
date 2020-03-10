#ifndef TENSORFLOW_SERVING_SERVABLES_TORCH_TORCH_TENSOR_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TORCH_TORCH_TENSOR_UTIL_H_

#include <torch/script.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "google/protobuf/text_format.h"

namespace tensorflow {
namespace serving {

bool torchTensorFromProto(torch::Tensor * tensor, const tensorflow::TensorProto &proto);
bool torchTensorToProto(const torch::Tensor *tensor, tensorflow::TensorProto* proto);
void PrintProtoToString(const protobuf::Message& message);

}  // namespace serving
}  // namespace tensorflow
#endif
