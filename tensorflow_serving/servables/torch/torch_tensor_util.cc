#include "tensorflow_serving/servables/torch/torch_tensor_util.h"
#include "tensorflow_serving/servables/torch/torch_types.h"

namespace tensorflow {
namespace serving {

template <typename T>
struct SerilazeHelper {
  typedef protobuf::RepeatedField<T> RepeatedFieldType;

  template<typename Source>
  static T* DecodeFromRawContent(const Source& in, int64 n){
    if (in.size() != sizeof(T) * n) {
      LOG(ERROR) << "LogUnexpectedSize: " << in.size() << " , " << sizeof(T) * n;
      return nullptr;
    }
    CHECK_GT(n, 0);
    T* out = new T[n];
    if (out == nullptr) {
      LOG(ERROR) << "allocate memory error";
      return nullptr;
    }
    memcpy(out, in.data(), in.size());
    return out;
  }
};

#define SINGLE_ARG(...) __VA_ARGS__
#define CASE_DECODE(TYPE, DECODE)           \
  case DataTypeToTorchEnum<TYPE>::value: {  \
    typedef TYPE T;                         \
    DECODE;                                 \
    break;                                  \
  }

#define CASES_DECODE(DType, DECODE)             \
  switch(DType){                                \
    CASE_DECODE(int, SINGLE_ARG(DECODE));       \
    CASE_DECODE(float, SINGLE_ARG(DECODE));     \
    CASE_DECODE(double, SINGLE_ARG(DECODE));    \
    default:                                    \
      break;                                    \
  }


template <typename T>
struct ProtoSerializeHelper {};

// For a C++ type "T" (float, double, int32, etc.), the repeated field
// "N"_val (float_val, int_val, label_val, etc.) of type "F" (float,
// int32, string, etc) in the TensorProto is used for serializing the
// tensor of type "T".
#define PROTO_TRAITS(T, F, N)                                          \
  template <>                                                          \
  struct ProtoSerializeHelper<T> {                                              \
    typedef SerilazeHelper<F>::RepeatedFieldType FieldType;                    \
    static FieldType::const_iterator Begin(const TensorProto& proto) { \
      return proto.N##_val().begin();                                  \
    }                                                                  \
    static size_t NumElements(const TensorProto& proto) {              \
      return proto.N##_val().size();                                   \
    }                                                                  \
    static void Fill(const T* data, size_t n, TensorProto* proto) {    \
      typename ProtoSerializeHelper<T>::FieldType copy(data, data + n);         \
      proto->mutable_##N##_val()->Swap(&copy);                         \
    }                                                                  \
  };
PROTO_TRAITS(uint8, int32, int);
PROTO_TRAITS(int8, int32, int);
PROTO_TRAITS(int16, int32, int);
PROTO_TRAITS(int32, int32, int);
PROTO_TRAITS(float, float, float);
PROTO_TRAITS(double, double, double);
#undef PROTO_TRAITS

template <>
struct ProtoSerializeHelper<int64> {
  static const int64* Begin(const TensorProto& proto) {
    return reinterpret_cast<const int64*>(proto.int64_val().begin());
  }
  static size_t NumElements(const TensorProto& proto) {
    return proto.int64_val().size();
  }
  static void Fill(const int64* data, size_t n, TensorProto* proto) {
    protobuf::RepeatedField<protobuf_int64> copy(data, data + n);
    proto->mutable_int64_val()->Swap(&copy);
  }
};

template <>
struct ProtoSerializeHelper<uint64> {
  static const uint64* Begin(const TensorProto& proto) {
    return reinterpret_cast<const uint64*>(proto.uint64_val().begin());
  }
  static size_t NumElements(const TensorProto& proto) {
    return proto.uint64_val().size();
  }
  static void Fill(const uint64* data, size_t n, TensorProto* proto) {
    protobuf::RepeatedField<protobuf_uint64> copy(data, data + n);
    proto->mutable_uint64_val()->Swap(&copy);
  }
};

template <typename T>
T* DecodeFromProtoField(const tensorflow::TensorProto& in, int64 n) {
  CHECK_GT(n, 0);
  T* data = new T[n];
  if (data == nullptr) {
    LOG(ERROR) << "allocate memory error";
    return nullptr;
  }
  const int64 in_n = ProtoSerializeHelper<T>::NumElements(in);
  if (in_n <= 0) {
    std::fill_n(data, n, T());
  } else {
    auto begin = ProtoSerializeHelper<T>::Begin(in);
    if (n <= in_n) {
      std::copy_n(begin, n, data);
    } else {
      std::copy_n(begin, in_n, data);
      const T& last = *(data + in_n - 1);
      std::fill_n(data + in_n, n - in_n, last);
    }
  }
  return data;
}

// Copies T[n] stored in the buffer "in" into the repeated field in
// "out" corresponding to type T.
template <typename T>
void EncodeToProtoField(const T* data, int64 n, TensorProto* out) {
  ProtoSerializeHelper<T>::Fill(data, n, out);
}

template <typename T>
void deleter(void *data){
  T*d = (T*)data;
  delete []d;
}


bool tensorshapeFromProto(int64 *n, std::vector<int64_t> *shapeVec, 
                          const tensorflow::TensorShapeProto& proto){
  if (proto.dim_size() > 0){
    auto numel = 1;
    shapeVec->reserve(proto.dim_size());
    for (auto dim : proto.dim()){
      numel *=  dim.size();
      shapeVec->push_back(dim.size());
    }
    *n = numel;
    return true;
  }
  return false;
}

bool torchTensorFromProto(torch::Tensor * tensor, const tensorflow::TensorProto &proto){
  if (!TensorShape::IsValid(proto.tensor_shape())){
    return false;
  }
  if (proto.dtype() == tensorflow::DT_INVALID){
    return false;
  }
  // 1. convert tensorShape
  int64 N = 0;
  std::vector<int64_t> shapeVec;
  if (!tensorshapeFromProto(&N, &shapeVec, proto.tensor_shape())){
    LOG(ERROR) << "tensorshapeFromProto error";
    return false;
  }
  at::IntArrayRef shape(shapeVec);
  // 2. convert dataType
  const torch::Dtype dtype = tensorflowDtypeToTorchDtype(proto.dtype());
  if (dtype == at::ScalarType::Undefined){
    LOG(ERROR) << "tensorflowDtypeToTorchDtype error, tfDtype: " << proto.dtype();
    return false;
  }
  torch::TensorOptions option(dtype);
  // 3. convert tensorData
  void* data = nullptr;
  if (N > 0) {
    if (!proto.tensor_content().empty()) {
      CASES_DECODE(dtype, data = SerilazeHelper<T>::DecodeFromRawContent(proto.tensor_content(), N));
    } else {
      CASES_DECODE(dtype, data = DecodeFromProtoField<T>(proto, N));
    }
    if (data == nullptr) return false;
  }
  CASES_DECODE(dtype, *tensor = torch::from_blob(data, shape, deleter<T>, option));
  return true;
}


bool tensorshapeToProto(const at::IntArrayRef &shape, TensorShapeProto* proto){
  proto->Clear();
  if (shape.size()<=0 || shape.data()[0]<=0) {
    proto->set_unknown_rank(true);
  } else {
    for (auto i = 0; i < shape.size(); i++) {
      proto->add_dim()->set_size(shape.data()[i]);
    }
  }
  return true;
}

bool torchTensorToProto(const torch::Tensor *tensor, tensorflow::TensorProto* proto){
  proto->Clear();
  if (!tensorshapeToProto(tensor->sizes(), proto->mutable_tensor_shape())){
    LOG(ERROR) << "tensorshapeToProto failed";
    return false;
  }
  tensorflow::DataType dtype = TorchDtypeTotensorflowDtype(tensor->scalar_type());
  if (dtype == tensorflow::DT_INVALID){
    LOG(ERROR) << "TorchDtypeTotensorflowDtype failed, torchDtype: " << tensor->scalar_type();
    return false;
  }
  proto->set_dtype(dtype);
  if (tensor->data_ptr()) {
     CASES_DECODE(tensor->scalar_type(), EncodeToProtoField<T>(tensor->data_ptr<T>(), tensor->numel(), proto));
  }
  return true;
}


void PrintProtoToString(const protobuf::Message& message){
  static string msg_str;
  bool ret = protobuf::TextFormat::PrintToString(message, &msg_str);
  if (!ret){
    LOG(ERROR) << "ProtoMessage Print Error";
    return;
  }
  LOG(INFO) << "ProtoMessage: " << msg_str;
}


}  // namespace serving
}  // namespace tensorflow
