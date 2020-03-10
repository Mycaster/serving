# Description: Tensorflow Serving torch servable.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "torch_source_adapter",
    srcs = ["torch_source_adapter.cc"],
    hdrs = ["torch_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":torch_source_adapter_proto",
        ":constants",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "@com_pytorch_org_libtorch//:libtorch",
    ],
    alwayslink = 1,
)

cc_library(
    name = "torch_types",
    hdrs = ["torch_types.h"],
    deps = [
        "@com_pytorch_org_libtorch//:libtorch", 
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "torch_tensor_util",
    srcs = ["torch_tensor_util.cc"],
    hdrs = ["torch_tensor_util.h"],
    deps = [
        ":torch_types",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf_archive//:protobuf",
    ],
)

cc_library(
    name = "torch_predict_impl",
    srcs = ["torch_predict_impl.cc"],
    hdrs = ["torch_predict_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":torch_tensor_util", 
        "//tensorflow_serving/apis:predict_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:server_core",
    ],
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "torch_source_adapter_proto",
    srcs = ["torch_source_adapter.proto"],
    cc_api_version = 2,
)


cc_binary(
    name = "torch_module_test",
    srcs = ["torch_module_test.cc"],
    deps = [
        "@com_pytorch_org_libtorch//:libtorch",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:lib",

    ],
    linkopts = ["-lpthread"],
    copts = ["-D_GLIBCXX_USE_CXX11_ABI=1"],
)
