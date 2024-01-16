#include "onnxruntime_c_api.h"
#include <stdlib.h>
#include <stdio.h>
#include <wchar.h>
#include <string.h>

#ifdef USE_DML
#include "dml_provider_factory.h"
#endif

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

__declspec(dllexport) const OrtApi* g_ort;
__declspec(dllexport) OrtEnv* env;
__declspec(dllexport) OrtSession* session;
__declspec(dllexport) OrtSessionOptions* session_options;
__declspec(dllexport) OrtAllocator* allocator;
__declspec(dllexport) const OrtMemoryInfo* memory_info;

__declspec(dllexport) int create_ort_session(char* model_path_char, int n_threads);
__declspec(dllexport) int release_all_globals();
__declspec(dllexport) int release_ort_tensor(void* tensor);
__declspec(dllexport) int create_tensor_float(float* tensor_data, size_t data_len, int64_t* shape, size_t shape_len, void** tensor);
__declspec(dllexport) int create_tensor_int32(int32_t* tensor_data, size_t data_len, int64_t* shape, size_t shape_len, void** tensor);
__declspec(dllexport) size_t get_input_count(void);
__declspec(dllexport) size_t get_output_count(void);
__declspec(dllexport) int print_tensor_int32(void* tensor);

__declspec(dllexport) int run_session(void** tensors, char* output_name, int** output_ids_raw, size_t* output_len);
__declspec(dllexport) int free_ptr(void* ptr);
