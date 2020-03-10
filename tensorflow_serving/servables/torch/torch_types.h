
#ifndef TENSORFLOW_SERVING_SERVABLES_TORCH_TORCH_TYPES_H_
#define TENSORFLOW_SERVING_SERVABLES_TORCH_TORCH_TYPES_H_

#include <torch/script.h>
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace serving {

#define TYPE_CONVERT_CASE(SRC_DTYPE, DEST_DTYPE) \
    case SRC_DTYPE: {                            \
      return DEST_DTYPE;                         \
    }

inline const torch::Dtype tensorflowDtypeToTorchDtype(tensorflow::DataType dtype) {
  switch (dtype) {
    TYPE_CONVERT_CASE(tensorflow::DT_UINT8, torch::kUInt8)
    TYPE_CONVERT_CASE(tensorflow::DT_INT8, torch::kInt8)
    TYPE_CONVERT_CASE(tensorflow::DT_INT16, torch::kInt16)
    TYPE_CONVERT_CASE(tensorflow::DT_INT32, torch::kInt32)
    TYPE_CONVERT_CASE(tensorflow::DT_INT64, torch::kInt64)
    TYPE_CONVERT_CASE(tensorflow::DT_HALF, torch::kFloat16)
    TYPE_CONVERT_CASE(tensorflow::DT_FLOAT, torch::kFloat32)
    TYPE_CONVERT_CASE(tensorflow::DT_DOUBLE, torch::kFloat64)
    default:
      return torch::kFloat32;
  }
}

inline const tensorflow::DataType TorchDtypeTotensorflowDtype(torch::Dtype dtype) {
  switch (dtype) {
    TYPE_CONVERT_CASE(torch::kUInt8, tensorflow::DT_UINT8)
    TYPE_CONVERT_CASE(torch::kInt8, tensorflow::DT_INT8)
    TYPE_CONVERT_CASE(torch::kInt16, tensorflow::DT_INT16)
    TYPE_CONVERT_CASE(torch::kInt32, tensorflow::DT_INT32)
    TYPE_CONVERT_CASE(torch::kInt64, tensorflow::DT_INT64)
    TYPE_CONVERT_CASE(torch::kFloat16, tensorflow::DT_HALF)
    TYPE_CONVERT_CASE(torch::kFloat32, tensorflow::DT_FLOAT)
    TYPE_CONVERT_CASE(torch::kFloat64, tensorflow::DT_DOUBLE)
    default:
      return tensorflow::DT_FLOAT;
  }
}

// Validates type T for whether it is a supported DataType.
template <class T>
struct IsValidTorchDataType;

// DataTypeToTorchEnum<T>::v() and DataTypeToTorchEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToTorchEnum<float>::v() is DT_FLOAT.
template <class T>
struct DataTypeToTorchEnum {
  static_assert(IsValidTorchDataType<T>::value, "Specified Data Type not supported");
};  // Specializations below


// TorchEnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// TorchEnumToDataType<DT_FLOAT>::Type is float.
template <torch::Dtype VALUE>
struct TorchEnumToDataType {};  // Specializations below

// Template specialization for both DataTypeToTorchEnum and TorchEnumToDataType.
#define MATCH_DTYPE_AND_TORCHENUM(TYPE, ENUM)           \
  template <>                                           \
  struct DataTypeToTorchEnum<TYPE> {                    \
    static constexpr torch::Dtype value = ENUM;         \
  };                                                    \
  template <>                                           \
  struct IsValidTorchDataType<TYPE> {                   \
    static constexpr bool value = true;                 \
  };                                                    \
  template <>                                           \
  struct TorchEnumToDataType<ENUM> {                    \
    typedef TYPE Type;                                  \
  }

MATCH_DTYPE_AND_TORCHENUM(tensorflow::int32, torch::kInt32);
MATCH_DTYPE_AND_TORCHENUM(int64_t, torch::kInt64);
MATCH_DTYPE_AND_TORCHENUM(float, torch::kFloat32);
MATCH_DTYPE_AND_TORCHENUM(double, torch::kFloat64);
#undef MATCH_DTYPE_AND_TORCHENUM


// All types not specialized are marked invalid.
template <class T>
struct IsValidTorchDataType {
  static constexpr bool value = false;
};

}  // namespace serving
}  // namespace tensorflow
#endif