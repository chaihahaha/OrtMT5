#include "utils.h"
#include "onnxruntime_cxx_api.h"
#include "sentencepiece_processor.h"
#include <vector>
#include <string>
#include <filesystem>
#include <new>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/ustream.h>

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
        std::vector<const char*> input_node_names;
        std::vector<const char*> output_node_names;

        int32_t max_length;
        int32_t min_length;
        int32_t num_beams;
        int32_t num_return_sequences;
        float length_penalty;
        float repetition_penalty;
        int max_threads;
        int num_input_nodes;
        int num_output_nodes;

        TranslatorSession(std::string model_path, std::string spm_tokenizer_path, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty);
        std::string translate(std::string input_raw);
        std::string decode(std::vector<int> translated_token_ids);
        std::vector<int> decode_ortvalue(Ort::Value& tensor);
        std::vector<int> encode(std::string input_raw);
};
extern "C"
{
    __declspec(dllexport) void* create_translator_session(char* model_path_char, char* spm_tokenizer_path_char, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty);
    __declspec(dllexport) int delete_translator_session(void* session);
    __declspec(dllexport) char* run_translate(void* ptr, char* in);
}
