#include "utils.h"
#include "onnxruntime_c_api.h"
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdlib.h>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/ustream.h>
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
    __declspec(dllexport) int create_ort_api(void);
    __declspec(dllexport) int create_ort_session(char* model_path_char);
    __declspec(dllexport) int release(void);
 
    __declspec(dllexport) int run_session(int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty, int* input_ids_raw, size_t input_len, int** output_ids_raw, size_t* output_len);
}
