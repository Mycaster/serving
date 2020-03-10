#!/bin/bash
/tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --model_config_file="/tensorflow-serving/scripts/torch/model/torchmodel.config" --platform_config_file="/tensorflow-serving/scripts/torch/model/torch_source_adpter.config"
