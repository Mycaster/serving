#include <torch/script.h>

#include <string>
#include <utility>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/torch/torch_predict_impl.h"
#include "tensorflow_serving/servables/torch/torch_source_adapter.h"
#include "tensorflow_serving/servables/torch/torch_tensor_util.h"

namespace tensorflow {
namespace serving {

Status RunTorchPredict(TorchScriptModule& module, 
        const PredictRequest& request, PredictResponse* response) {
  std::vector<torch::jit::IValue> inputs;
  inputs.reserve(request.inputs().size());
  for (auto& input : request.inputs()) {
    const string & alias = input.first;
    torch::Tensor tensor;
    if (!torchTensorFromProto(&tensor, input.second)) {
      LOG(ERROR) << "tensor parsing error:" << alias;
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "tensor parsing error: " + alias);
    }
    LOG(INFO) << "tensor.numel: " << tensor.numel() << " size: " << tensor.sizes();
    //LOG(INFO) << "inputtensor: " << tensor.slice(0, 0, 1);
    inputs.push_back(tensor);
  }
  LOG(INFO) << "start forword...";
  auto ivalue = module.forward(inputs);
  /* only support toTensor() and toTuple() for now */
  if (ivalue.isTensor()){
    auto output = ivalue.toTensor();
    if (!torchTensorToProto(&output, &((*response->mutable_outputs())["output"]))) {
      LOG(ERROR) << "torchTensorToProto error:";
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                  "torchTensorToProto error: ");
    }
  }else if (ivalue.isTuple()){
    auto outputs = ivalue.toTuple();
    for (auto i=0; i < outputs->elements().size(); i++){
      string name = "output" + i;
      auto output = outputs->elements()[i].toTensor();
      if (!torchTensorToProto(&output, &((*response->mutable_outputs())[name]))) {
        LOG(ERROR) << "torchTensorToProto error:";
        return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                    "torchTensorToProto error: ");
      }
    }
  }else {
    LOG(ERROR) << "ivalue::Tag error, not Tensor or Tuple";
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "ivalue::Tag error, only support toTensor() and toTuple() for now");
  }
  PrintProtoToString(*response);
  return Status::OK();
}

Status PytorchPredictor::Predict(ServerCore* core, 
                                 const PredictRequest& request,
                                 PredictResponse* response) {
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }
  ServableHandle<TorchScriptModuleBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(request.model_spec(), &bundle));
  return RunTorchPredict(bundle->module_, request, response);
}

}  // namespace serving
}  // namespace tensorflow
