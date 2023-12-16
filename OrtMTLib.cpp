#include "OrtMTLib.h"


extern "C"
{
    __declspec(dllexport) int create_ort_api(const void** ort_api_py)
    {
        const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        *ort_api_py = g_ort;
        return 0;
    }
    __declspec(dllexport) int create_ort_session(const void* g_ort_py, char* model_path_char, POINTER_c_char_p* input_names, size_t* num_input_nodes, POINTER_c_char_p* output_names, size_t* num_output_nodes, void** env_py, void** session_py)
    {
        const OrtApi* g_ort = reinterpret_cast<const OrtApi*>(g_ort_py);
        std::string model_path(model_path_char);

        OrtEnv* env;
        ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "OrtMTLib", &env));
        OrtSessionOptions* session_options;
        ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

#ifdef _WIN32
        std::wstring model_path_wstring = str2wstr(model_path);
        const wchar_t* model_path_wchar = model_path_wstring.c_str();
#endif

        if (!std::filesystem::exists(model_path))
        {
            throw "Incorrect model path";
        }

        OrtSession* session;
#ifdef _WIN32
        ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path_wchar, session_options, &session));
#else
        ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path.c_str(), session_options, &session));
#endif
        *session_py = session;
        *env_py = env;
        //ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, num_input_nodes));
        //ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, num_output_nodes));

        //OrtAllocator* allocator;
        //ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

        //std::vector<char*> input_names_list;
        //// iterate over all input nodes
        //for (size_t i = 0; i < *num_input_nodes; i++)
        //{
        //    char* input_name;
        //    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &input_name));
        //    input_names_list.push_back(input_name);
        //}
        //*input_names = input_names_list.data();

        //std::vector<char*> output_names_list;
        //// iterate over all input nodes
        //for (size_t i = 0; i < *num_output_nodes; i++)
        //{
        //    char* output_name;
        //    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &output_name));
        //    output_names_list.push_back(output_name);
        //}
        //*output_names = output_names_list.data();
        return 0;
    }

    __declspec(dllexport) int create_sp_tokenizer(char* spm_tokenizer_path_char, void** sp_py)
    {
        std::string spm_tokenizer_path(spm_tokenizer_path_char);
        if (!std::filesystem::exists(spm_tokenizer_path))
        {
            throw "Incorrect tokenizer path";
        }
        sentencepiece::SentencePieceProcessor* sp = new sentencepiece::SentencePieceProcessor();
        const auto spm_status = sp->Load(spm_tokenizer_path);
        *sp_py = sp;
        return 0;
    }

    __declspec(dllexport) void encode_as_ids(void* sp_py, char* input_char, int** token_ids, size_t* n_tokens)
    {
        sentencepiece::SentencePieceProcessor* sp = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(sp_py);
        std::string input_raw(input_char);
        std::vector<int> input_ids = sp->EncodeAsIds(input_raw);
        *token_ids = input_ids.data();
        *n_tokens = input_ids.size();
        //std::cout << "encode to ids" << std::endl;
        //for (int ii = 0; ii < input_ids.size(); ii++)
        //{
        //    std::cout << input_ids[ii] << "," << std::endl;
        //}
        //std::cout << std::endl;
        return;
    }

    __declspec(dllexport) void decode_from_ids(void* sp_py, int* output_ids_raw, size_t n_tokens, char** output_char)
    {
        sentencepiece::SentencePieceProcessor* sp = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(sp_py);
        std::vector<int> output_ids(output_ids_raw, output_ids_raw + n_tokens);
        std::string translated_str;
        sp->Decode(output_ids, &translated_str);
        *output_char = (char*) translated_str.c_str();
        return;
    }
     


    //__declspec(dllexport) int delete_ptr(void* ptr)
    //{
    //    delete ptr;
    //    return 0;
    //}

    __declspec(dllexport) int run_session(const void* g_ort_py, void* session_py, int max_length, int min_length, int num_beams, int num_return_sequences, float length_penalty, float repetition_penalty, int* input_ids_raw, size_t input_len, int** output_ids_raw, size_t* output_len)
    {
        const OrtApi* g_ort = reinterpret_cast<const OrtApi*>(g_ort_py);
        OrtSession* session = reinterpret_cast<OrtSession*>(session_py);
        OrtAllocator* allocator;
        ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));
        const OrtMemoryInfo* memory_info;
        //ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        ORT_ABORT_ON_ERROR(g_ort->AllocatorGetInfo(allocator, &memory_info));

        std::vector<int64_t> one = {1};

        OrtValue* max_length_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &max_length, sizeof(int32_t),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &max_length_tensor));
        
        OrtValue* min_length_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &min_length, sizeof(int32_t),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &min_length_tensor));
        
        OrtValue* num_beams_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &num_beams, sizeof(int32_t),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &num_beams_tensor));
        
        OrtValue* num_return_sequences_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &num_return_sequences, sizeof(int32_t),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &num_return_sequences_tensor));
        
        OrtValue* length_penalty_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &length_penalty, sizeof(float),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &length_penalty_tensor));
        
        OrtValue* repetition_penalty_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                &repetition_penalty, sizeof(float),
                one.data(), 1,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &repetition_penalty_tensor));
        
        std::vector<int> input_ids(input_ids_raw, input_ids_raw + input_len);
        std::vector<int64_t> input_ids_dims{1, (int64_t)input_ids.size()};

        OrtValue* input_ids_tensor = NULL;
        ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
                memory_info,
                input_ids.data(), sizeof(int32_t) * input_ids.size(),
                input_ids_dims.data(),1 * input_ids_dims.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &input_ids_tensor));
        
        std::vector<OrtValue*> input_tensors;
        input_tensors.push_back(input_ids_tensor);
        input_tensors.push_back(max_length_tensor);
        input_tensors.push_back(min_length_tensor);
        input_tensors.push_back(num_beams_tensor);
        input_tensors.push_back(num_return_sequences_tensor);
        input_tensors.push_back(length_penalty_tensor);
        input_tensors.push_back(repetition_penalty_tensor);

        //g_ort->ReleaseMemoryInfo(memory_info);

        ////////// copied from create_session
        size_t num_input_nodes, num_output_nodes;
        ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &num_input_nodes));
        ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &num_output_nodes));

        std::vector<char*> input_names_list;
        // iterate over all input nodes
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            char* input_name;
            ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &input_name));
            input_names_list.push_back(input_name);
        }
        //char** input_names = input_names_list.data();

        std::vector<char*> output_names_list;
        // iterate over all input nodes
        for (size_t i = 0; i < num_output_nodes; i++)
        {
            char* output_name;
            ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &output_name));
            output_names_list.push_back(output_name);
        }
        //char** output_names = output_names_list.data();
        /////////////////copy end

        std::vector<int64_t> output_dims = {1, 1, max_length};
        OrtValue* output_tensor = NULL;
        
        ORT_ABORT_ON_ERROR(g_ort->Run(
                session, NULL,
                input_names_list.data(), input_tensors.data(), input_names_list.size(),
                output_names_list.data(), output_names_list.size(), &output_tensor
                ));

        OrtTensorTypeAndShapeInfo* info;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensor, &info));
        ORT_ABORT_ON_ERROR(g_ort->GetTensorShapeElementCount(info, output_len));
        int* output_ids;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_ids));
        *output_ids_raw = output_ids;

        for (int i = 0; i < input_tensors.size(); i++)
        {
            g_ort->ReleaseValue(input_tensors[i]);
        }
        g_ort->ReleaseValue(output_tensor);
        std::cout << "released values" << std::endl;
        //g_ort->ReleaseAllocator(allocator);
        g_ort->ReleaseTensorTypeAndShapeInfo(info);
        return 0;
    }
}
