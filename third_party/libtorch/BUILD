# libtorch (libtorch.org) library.
# from https://github.com/libtorch/libtorch

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE"])

TORCH_HEADERS = glob([
    "include/**/*.h",
])

TORCH_LIBRARIES = [
    "lib/libtorch.so",
    "lib/libc10_cuda.so",
    "lib/libc10.so",
    "lib/libcudart.so.10.1",
    "lib/libcudart-1b201d85.so.10.1",
    "lib/libnvToolsExt-3965bdd0.so.1",
    "lib/libgomp-753e6e92.so.1",
]
cc_library(
    name = "libtorch",
    srcs = TORCH_LIBRARIES,
    hdrs = TORCH_HEADERS,
    includes = ["include", "include/torch/csrc/api/include"],
)

