#include "utils.h"
#include "onnxruntime_c_api.h"
#include "sentencepiece_processor.h"
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

//std::vector<int> decode_ortvalue(Ort::Value& tensor);
extern "C"
{
    __declspec(dllexport) int create_ort_api(const void** ort_api_py);
    __declspec(dllexport) int create_ort_session(const void* g_ort_py, char* model_path_char, POINTER_c_char_p* input_names, size_t* num_input_nodes, POINTER_c_char_p* output_names, size_t* num_output_nodes, void** env_py, void** session_py);
    __declspec(dllexport) int create_sp_tokenizer(char* spm_tokenizer_path_char, void** sp_py);
    __declspec(dllexport) void encode_as_ids(void* sp_py, char* input_char, int** token_ids, size_t* n_tokens);
    __declspec(dllexport) void decode_from_ids(void* sp_py, int* output_ids_raw, size_t n_tokens, char** output_char);
    __declspec(dllexport) int delete_ptr(void* ptr);
    __declspec(dllexport) int run_session(const void* g_ort_py, void* session_py, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty, int* input_ids_raw, size_t input_len, int** output_ids_raw, size_t* output_len);
 
}
