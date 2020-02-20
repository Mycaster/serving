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
    name = "torch_source_adapter",
    srcs = ["torch_source_adapter.cc"],
    hdrs = ["torch_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":torch_source_adapter_proto",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "@com_pytorch_org_libtorch//:libtorch",
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
    ],
    copts = ["-D_GLIBCXX_USE_CXX11_ABI=1"],
)
