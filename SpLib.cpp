#include "SpLib.h"

extern "C"
{
    sentencepiece::SentencePieceProcessor* sp = nullptr;
    __declspec(dllexport) int create_sp_tokenizer(char* spm_tokenizer_path_char)
    {
        std::string spm_tokenizer_path(spm_tokenizer_path_char);
        if (!std::filesystem::exists(spm_tokenizer_path))
        {
            std::cout << "Incorrect tokenizer path" << std::endl;;
            return -1;
        }
        sp = new sentencepiece::SentencePieceProcessor();
        const auto spm_status = sp->Load(spm_tokenizer_path);
        return 0;
    }

    __declspec(dllexport) int encode_as_ids(char* input_char, int** token_ids, size_t* n_tokens)
    {
        std::string input_raw(input_char);
        std::vector<int> input_ids = sp->EncodeAsIds(input_raw);
        size_t input_ids_memlen = input_ids.size() * sizeof(int);
        *token_ids = (int*)malloc(input_ids_memlen);
        memcpy(*token_ids, &input_ids.front(), input_ids_memlen);
        *n_tokens = input_ids.size();
        return 0;
    }

    __declspec(dllexport) int decode_from_ids(int* output_ids_raw, size_t n_tokens, char** output_char)
    {
        std::vector<int> output_ids(output_ids_raw, output_ids_raw + n_tokens);
        std::string translated_str;
        sp->Decode(output_ids, &translated_str);
        //*output_char = (char*) translated_str.c_str();
        size_t strlen = translated_str.length();
        *output_char = new char[strlen + 1];
        strcpy(*output_char, translated_str.c_str());
        return 0;
    }
    


}
