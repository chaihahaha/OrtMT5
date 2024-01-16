#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <filesystem>
#include "sentencepiece_processor.h"

extern "C"
{
    __declspec(dllexport) int create_sp_tokenizer(char* spm_tokenizer_path_char);
    __declspec(dllexport) int encode_as_ids(char* input_char, int** token_ids, size_t* n_tokens);
    __declspec(dllexport) int decode_from_ids(int* output_ids_raw, size_t n_tokens, char** output_char);
    __declspec(dllexport) int free_ptr(void* ptr);
}
