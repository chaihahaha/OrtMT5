#include "windows.h"
#include "onnxruntime_c_api.h"
#include <stdlib.h>
#include <stdio.h>
//#include <nlohmann/json.hpp>
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

typedef char** POINTER_c_char_p;

extern "C"
{
extern    __declspec(dllexport) const OrtApi* g_ort;
extern    __declspec(dllexport) OrtEnv* env;
extern    __declspec(dllexport) OrtSession* session;
extern    __declspec(dllexport) OrtSessionOptions* session_options;
extern    __declspec(dllexport) OrtAllocator* allocator;
extern    __declspec(dllexport) OrtMemoryInfo* memory_info;

    __declspec(dllexport) int WINAPI create_ort_api(void);
    __declspec(dllexport) int WINAPI create_ort_session(char* model_path_char);
    __declspec(dllexport) int WINAPI release(void);
 
    __declspec(dllexport) int WINAPI run_session(int32_t max_length, int32_t min_length, int32_t num_beams, int32_t num_return_sequences, float length_penalty, float repetition_penalty, int32_t* input_ids_raw, size_t input_len, int** output_ids_raw, size_t* output_len);
    __declspec(dllexport) int run_session_decoder(void);
}
