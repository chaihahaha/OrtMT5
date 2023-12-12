#include "OrtMTLib.h"

std::vector<int> decode_ortvalue(Ort::Value& tensor)
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


extern "C"
{
    __declspec(dllexport) void* create_ort_session(char* model_path_char)
    {
        std::string model_path(model_path_char);
        Ort::SessionOptions session_options;
        //max_threads = std::thread::hardware_concurrency();

        //session_options.SetIntraOpNumThreads(max_threads/2);
        //session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        //session_options.SetInterOpNumThreads(max_threads/2);
        //session_options.SetGraphOptimizationLevel(
        //    GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "OrtMT5");

#ifdef _WIN32
        std::wstring model_path_wstring = str2wstr(model_path);
        const wchar_t* model_path_wchar = model_path_wstring.c_str();
#endif

        if (!std::filesystem::exists(model_path))
        {
            throw "Incorrect model path";
        }

#ifdef _WIN32
        return new Ort::Session(env, model_path_wchar, session_options);
#else
        return new Ort::Session(env, model_path.c_str(), session_options);
#endif
    }

    __declspec(dllexport) void* create_sp_tokenizer(char* spm_tokenizer_path_char)
    {
        std::string spm_tokenizer_path(spm_tokenizer_path_char);
        if (!std::filesystem::exists(spm_tokenizer_path))
        {
            throw "Incorrect tokenizer path";
        }
        sentencepiece::SentencePieceProcessor* sp = new sentencepiece::SentencePieceProcessor();
        const auto spm_status = sp->Load(spm_tokenizer_path);
        return sp;
    }


    __declspec(dllexport) int delete_ptr(void* ptr)
    {
        delete ptr;
        return 0;
    }
    __declspec(dllexport) char* run_translate(void* session_py, void* sp_py, char* in, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty)
    {
        Ort::Session* session = reinterpret_cast<Ort::Session*>(session_py);
        sentencepiece::SentencePieceProcessor* sp = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(sp_py);

        std::string input_raw(in);
        std::cout << "starting translate" << std::endl;
        std::vector<int> input_ids = sp->EncodeAsIds(input_raw);
        std::cout << "encode to ids" << std::endl;
        for (int ii = 0; ii < input_ids.size(); ii++)
        {
            std::cout << input_ids[ii] << "," << std::endl;
        }
        std::cout << std::endl;
        std::vector<int64_t> one = {1};
        
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::ConstMemoryInfo memory_info = allocator.GetInfo();
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
        std::cout << "consts:" << max_length << " " << min_length << " " << num_beams << " " << num_return_sequences << " " << length_penalty << " " << repetition_penalty << " " << std::endl;


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

        std::vector<char*> input_node_names, output_node_names;
        size_t num_input_nodes = session->GetInputCount();
        input_node_names.reserve(num_input_nodes);
        std::cout << "get input count" << std::endl;

        // iterate over all input nodes
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
        }
        std::cout << "get input names" << std::endl;

        size_t num_output_nodes = session->GetOutputCount();
        output_node_names.reserve(num_output_nodes);

        // iterate over all output nodes
        for (size_t i = 0; i < num_output_nodes; i++)
        {
            Ort::AllocatedStringPtr output_name = session->GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(output_name.get());
        }

        std::cout << "prepared input name: " << input_node_names[0] << " " << output_node_names.size() << std::endl;
        std::cout << "prepared output name: " << output_node_names[0] << " " << output_node_names.size() << std::endl;

        std::vector<Ort::Value> output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_node_names.data(),
            output_node_names.size()
            );
        std::cout << "run session" << std::endl;
        std::vector<int> output_ids = decode_ortvalue(output_tensors[0]);
        std::string translated_str;
        sp->Decode(output_ids, &translated_str);
        char* out = const_cast<char*>(translated_str.c_str());
        return out;
    }
}
