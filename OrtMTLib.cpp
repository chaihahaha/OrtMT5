#include "OrtMTLib.h"

#pragma comment(lib, "onnxruntime.lib")

TranslatorSession::TranslatorSession(std::string model_path, std::string spm_tokenizer_path, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty)
{
    if (!std::filesystem::exists(spm_tokenizer_path))
    {
        throw "Incorrect tokenizer path";
    }
    sp = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto spm_status = sp->Load(spm_tokenizer_path);

    this->max_length = max_length;
    this->min_length = min_length;
    this->num_beams = num_beams;
    this->num_return_sequences = num_return_sequences;
    this->length_penalty = length_penalty;
    this->repetition_penalty = repetition_penalty;

    Ort::SessionOptions session_options;
    max_threads = std::thread::hardware_concurrency();

    session_options.SetIntraOpNumThreads(max_threads/2);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetInterOpNumThreads(max_threads/2);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    this->env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "OrtMT5");

#ifdef _WIN32
    std::wstring model_path_wstring = str2wstr(model_path);
    const wchar_t* model_path_wchar = model_path_wstring.c_str();
#endif

    if (!std::filesystem::exists(model_path))
    {
        throw "Incorrect model path";
    }

#ifdef _WIN32
    this->session = std::make_unique<Ort::Session>(this->env, model_path_wchar, session_options);
#else
    this->session = std::make_unique<Ort::Session>(this->env, model_path.c_str(), session_options);
#endif

    num_input_nodes = session->GetInputCount();
    this->input_node_names.reserve(num_input_nodes);

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++)
    {
        Ort::AllocatedStringPtr input_name = this->session->GetInputNameAllocated(i, this->allocator);
        this->input_node_names.push_back(input_name.get());
    }

    num_output_nodes = session->GetOutputCount();
    this->output_node_names.reserve(num_output_nodes);

    // iterate over all input nodes
    for (size_t i = 0; i < num_output_nodes; i++)
    {
        Ort::AllocatedStringPtr output_name = this->session->GetOutputNameAllocated(i, this->allocator);
        this->output_node_names.push_back(output_name.get());
    }

    this->memory_info = this->allocator.GetInfo();

}

std::string TranslatorSession::decode(std::vector<int> translated_token_ids)
{
    std::string translated;
    this->sp->Decode(translated_token_ids, &translated);
    return translated;
}

std::vector<int> TranslatorSession::decode_ortvalue(Ort::Value& tensor)
{
    Ort::TensorTypeAndShapeInfo ts = tensor.GetTensorTypeAndShapeInfo();

    const int32_t* el = tensor.GetTensorData<int32_t>();
    std::vector<int> ids;
    for (size_t i = 0; i < ts.GetElementCount(); i++)
    {
        ids.push_back((int)el[i]);
    }
    return ids;
}

std::vector<int> TranslatorSession::encode(std::string input_raw)
{
    icu::UnicodeString input_us(input_raw.c_str());
    std::string input_str;
    input_us.toUTF8String(input_str);

    std::vector<int> input_ids = this->sp->EncodeAsIds(input_str);
    return input_ids;
}

std::string TranslatorSession::translate(std::string input_raw)
{
    std::vector<int> input_ids = this->encode(input_raw);
    //for (int ii = 0; ii < input_ids.size(); ii++)
    //{
    //    std::cout << input_ids[ii] << "," << std::endl;
    //}
    //std::cout << std::endl;
    std::vector<int64_t> one = {1};
    
    Ort::Value max_length_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, &max_length, 1,
            one.data(), 1);
    Ort::Value min_length_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, &min_length, 1,
            one.data(), 1);
    Ort::Value num_beams_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, &num_beams, 1,
            one.data(), 1);
    Ort::Value num_return_sequences_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, &num_return_sequences, 1,
            one.data(), 1);
    Ort::Value length_penalty_tensor = Ort::Value::CreateTensor<float>(memory_info, &length_penalty, 1,
            one.data(), 1);
    Ort::Value repetition_penalty_tensor = Ort::Value::CreateTensor<float>(memory_info, &repetition_penalty, 1,
            one.data(), 1);


    std::vector<int64_t> input_ids_dims{1, (int64_t)input_ids.size()};

    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, input_ids.data(), input_ids.size(),
        input_ids_dims.data(), input_ids_dims.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_ids_tensor));
    input_tensors.push_back(std::move(max_length_tensor));
    input_tensors.push_back(std::move(min_length_tensor));
    input_tensors.push_back(std::move(num_beams_tensor));
    input_tensors.push_back(std::move(num_return_sequences_tensor));
    input_tensors.push_back(std::move(length_penalty_tensor));
    input_tensors.push_back(std::move(repetition_penalty_tensor));

    std::vector<Ort::Value> output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_node_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_node_names.data(),
        output_node_names.size()
        );
    std::string translated_str = decode(decode_ortvalue(output_tensors[0]));
    return translated_str;
}

extern "C"
{
    __declspec(dllexport) void* create_translator_session(char* model_path_char, char* spm_tokenizer_path_char, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty)
    {
        std::string model_path(model_path_char);
        std::string spm_tokenizer_path(spm_tokenizer_path_char);
        return new(std::nothrow) TranslatorSession(model_path, spm_tokenizer_path, max_length, min_length, num_beams, num_return_sequences, length_penalty, repetition_penalty);
    }
    __declspec(dllexport) int delete_translator_session(void* session)
    {
        delete session;
        return 0;
    }
    __declspec(dllexport) char* run_translate(void* ptr, char* in)
    {
        TranslatorSession* session_ptr = reinterpret_cast<TranslatorSession*>(ptr);
        std::string in_str(in);
        std::string out_str = session_ptr->translate(in_str);
        char* out = const_cast<char*>(out_str.c_str());
        return out;
    }
}
