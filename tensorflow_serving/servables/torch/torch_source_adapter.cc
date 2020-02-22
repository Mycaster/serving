/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include <stddef.h>
#include <memory>
#include <vector>

#include "torch/script.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/servables/torch/torch_source_adapter.h"

namespace tensorflow {
namespace serving {

// using TorchModule = std::unordered_map<string, string>;
using TorchModule = torch::jit::script::Module;


class TorchBundle {
public:
  TorchBundle(TorchModule module): module_(module){};
  ~TorchBundle(){};
private:
  TorchModule module_;
};

TorchSourceAdapter::TorchSourceAdapter() {}

TorchSourceAdapter::~TorchSourceAdapter() { Detach(); }

Status TorchSourceAdapter::Convert(const StoragePath& path,
                                std::unique_ptr<Loader>* loader) {
  auto servable_creator = [path](std::unique_ptr<TorchBundle>* bundle) {
    LOG(INFO) << "path: " << path;
    torch::Tensor tensor = torch::eye(3);
    LOG(INFO) << tensor;
    ///* test module
    torch::jit::script::Module module;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load("/libtorch/torch-models/resnet/1/traced_resnet_model.pt");
    }
    catch (const c10::Error& e) {
      LOG(ERROR) << "error loading the model";
      return errors::Internal("error loading the model");
    }
    LOG(INFO) << "ok";
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));
    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    LOG(INFO) << output.slice(1, 0, 5) << '\n';
    //*/

    // TorchModule module = torch::jit::load(path);
    // bundle->reset(new TorchBundle(module));
    return Status::OK();
  };
  auto resource_estimator = [path](ResourceAllocation* estimate) {
    return Status::OK();
  };
  loader->reset(new SimpleLoader<TorchBundle>(servable_creator, resource_estimator));
  return Status::OK();
}

// Register the source adapter.
class TorchSourceAdapterCreator {
 public:
  static Status Create(
      const TorchSourceAdapterConfig& config,
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
          adapter) {
    LOG(INFO) << "TorchSourceAdapterCreator Create TorchSourceAdapter";
    adapter->reset(new TorchSourceAdapter());
    return Status::OK();
  }
};

REGISTER_STORAGE_PATH_SOURCE_ADAPTER(TorchSourceAdapterCreator,
                                     TorchSourceAdapterConfig);

}  // namespace serving
}  // namespace tensorflow
