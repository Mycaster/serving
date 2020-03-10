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

#ifndef TENSORFLOW_SERVING_SERVABLES_TORCH_TORCH_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_SERVABLES_TORCH_TORCH_SOURCE_ADAPTER_H_

#include "torch/script.h"

#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/torch/torch_source_adapter.pb.h"

namespace tensorflow {
namespace serving {

using TorchScriptModule = torch::jit::script::Module;
class TorchScriptModuleBundle {
public:
  explicit TorchScriptModuleBundle(TorchScriptModule &module)
      : module_(module){};
  ~TorchScriptModuleBundle(){};
  //TorchScriptModule get(){ return module_};

//private:
  TorchScriptModule module_;
};

class TorchSourceAdapter final
    : public UnarySourceAdapter<StoragePath, std::unique_ptr<Loader>> {
 public:
  TorchSourceAdapter();
  ~TorchSourceAdapter() override;

 private:
  friend class TorchSourceAdapterCreator;
  Status Convert(const StoragePath& path,
                 std::unique_ptr<Loader>* loader) override;

  TF_DISALLOW_COPY_AND_ASSIGN(TorchSourceAdapter);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TORCH_TORCH_SOURCE_ADAPTER_H_
