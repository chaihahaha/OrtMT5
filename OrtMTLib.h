#include "utils.h"
#include "onnxruntime_cxx_api.h"
#include "sentencepiece_processor.h"
#include <vector>
#include <string>
#include <filesystem>
#include <new>
#include <iostream>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/ustream.h>
std::vector<int> decode_ortvalue(Ort::Value& tensor);
extern "C"
{
    __declspec(dllexport) void* create_ort_session(char* model_path_char);
    __declspec(dllexport) void* create_sp_tokenizer(char* spm_tokenizer_path_char);
    __declspec(dllexport) int delete_ptr(void* ptr);
    __declspec(dllexport) char* run_translate(void* session_py, void* sp_py, char* in, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty);
 
}
