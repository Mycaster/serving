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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/servables/torch/constants.h"
#include "tensorflow_serving/servables/torch/torch_source_adapter.h"

namespace tensorflow {
namespace serving {

TorchSourceAdapter::TorchSourceAdapter() {}

TorchSourceAdapter::~TorchSourceAdapter() { Detach(); }

Status TorchSourceAdapter::Convert(const StoragePath& path,
                                std::unique_ptr<Loader>* loader) {
  auto servable_creator = [path](std::unique_ptr<TorchScriptModuleBundle>* bundle) {
    LOG(INFO) << "Reading TracedModel from dir: " << path;
    const string traced_model_path = io::JoinPath(path, kPytorchTracedModelFilenamePt);
    if (!Env::Default()->FileExists(traced_model_path).ok()) {
      return Status(error::Code::NOT_FOUND,
                    "Could not find PytorchTracedModel .pt at supplied export directory path: " +
                    path);
    }
    TorchScriptModule module;
    try{
      module = torch::jit::load(traced_model_path);
    } catch (const c10::Error& e) {
      return Status(error::Code::NOT_FOUND,
                    "Could not load PytorchTracedModel");
    }
    bundle->reset(new TorchScriptModuleBundle(module));
    return Status::OK();
  };
  auto resource_estimator = [path](ResourceAllocation* estimate) {
    return Status::OK();
  };
  loader->reset(new SimpleLoader<TorchScriptModuleBundle>(servable_creator, resource_estimator));
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
