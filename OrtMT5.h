#include "onnxruntime_cxx_api.h"
#include "sentencepiece_processor.h"
#include "argparse/argparse.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <atomic>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/ustream.h>

void async_callback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status_ptr);
void print_translated(Ort::Value& tensor);
void clear_ortvalue_vector(OrtValue** a, size_t len);
std::wstring str2wstr(std::string s);

class TranslatorSession
{
    public:
        std::unique_ptr<Ort::Session> session = nullptr;
        std::unique_ptr<sentencepiece::SentencePieceProcessor> sp = nullptr;

        Ort::Env env;
        Ort::ConstMemoryInfo memory_info;
        Ort::AllocatorWithDefaultOptions allocator;

        std::wstring model_path;
        std::string spm_tokenizer_path;
        std::vector<std::string> input_node_names;

        int max_length;
        int min_length;
        int num_beams;
        int num_return_sequences;
        float length_penalty;
        float repetition_penalty;

        TranslatorSession(std::string model_path, std::string spm_tokenizer_path, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty);
        std::string translate(std::string input_raw);
};
